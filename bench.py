#!/usr/bin/env python3

import os
import sys
import time
import subprocess
import gc
import statistics
import json
import threading
import re
import csv

# Need to avoid as much extra CPU usage as possible
gc.disable()

# sysfs power supply nodes for power sampling
POWER_SUPPLY = None
POWER_SUPPLY_NODES = [
    # Qualcomm Battery Management System + fuel gauge: preferred when available for more info
    "/sys/class/power_supply/bms",
    # Most common
    "/sys/class/power_supply/battery",
    # Nexus 10
    "/sys/class/power_supply/ds2784-fuelgauge",
]
# Some fuel gauges need current unit scaling
POWER_CURRENT_FACTOR = 1
POWER_CURRENT_NODES = [
    # Exynos devices with Maxim PMICs report µA separately
    "batt_current_ua_now",
    # Standard µA node
    "current_now",
]
# Full paths to final nodes
POWER_CURRENT_NODE = None
POWER_VOLTAGE_NODE = None

# Default power sampling intervals
POWER_SAMPLE_INTERVAL = 1000  # ms
POWER_SAMPLE_FG_DEFAULT_INTERVALS = {
    # qgauge updates every 100 ms, but sampling also uses power, so do it conservatively
    "qpnp,qg": 250,
    # qpnp-fg-gen3/4 update every 1000 ms
    "qpnp,fg": 1000,
    # SM8350+ aDSP fuel gauge updates every 1000 ms
    "qcom,pmic_glink": 1000,
}

# Needs to match init and cmdline
HOUSEKEEPING_CPU = 0

# cpu0 is for housekeeping, so we can't benchmark it
# Benchmark cpu1 instead, which is also in the little cluster
REPLACE_CPUS = {
    HOUSEKEEPING_CPU: 1,
}

# How long to idle at each freq and measure power before benchmarking
FREQ_IDLE_TIME = 5  # sec

# To reduce chances of an array realloc + copy during benchmark runs
PREALLOC_SECONDS = 300  # seconds of power sampling

# CoreMark PERFORMANCE_RUN params with 600,000 iterations
COREMARK_ITERATIONS = 600000
COREMARK_PERFORMANCE_RUN = ["0x0", "0x0", "0x66", str(COREMARK_ITERATIONS), "7", "1", "2000"]

# Blank lines are for rounded corner & camera cutout protection
BANNER = """



  __                _                     _     
 / _|_ __ ___  __ _| |__   ___ _ __   ___| |__  
| |_| '__/ _ \/ _` | '_ \ / _ \ '_ \ / __| '_ \ 
|  _| | |  __/ (_| | |_) |  __/ | | | (__| | | |
|_| |_|  \___|\__, |_.__/ \___|_| |_|\___|_| |_|
                 |_|                            

           CPU benchmark • by kdrag0n

------------------------------------------------
"""

SYS_CPU = "/sys/devices/system/cpu"

# Calculate prealloc slots now that the interval is known
PREALLOC_SLOTS = int(PREALLOC_SECONDS / (POWER_SAMPLE_INTERVAL / 1000))

_stop_power_mon = False
_prealloc_samples = [-1] * PREALLOC_SLOTS
_power_samples = _prealloc_samples

def pr_debug(*args, **kwargs):
    if __debug__:
        kwargs["flush"] = True
        print(*args, **kwargs)

def run_cmd(args):
    pr_debug(f"Running command: {args}")
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    pr_debug(f"Command exited with return code {proc.returncode}")
    if proc.returncode == 0:
        return proc.stdout
    else:
        raise ValueError(f"Subprocess {args} failed with exit code {proc.returncode}:\n{proc.stdout}")

def sample_energy(cluster):
    # Read out the energy used by this specific cluster as reported by the PMIC
    meter_pattern = r'.*T=(\d+).*' + r'VDD_CPUCL' + str(cluster) + r'], ' + r'(\d+)'
    power_data = read_file("/sys/bus/iio/devices/iio:device1/energy_value")
    result = re.search(meter_pattern, power_data)
    ms = int(result.group(1))
    uj = int(result.group(2))
    return ms, uj

def write_cpu(cpu, node, content):
    pr_debug(f"Writing CPU value: cpu{cpu}/{node} => {content}")
    with open(f"{SYS_CPU}/cpu{cpu}/{node}", "w") as f:
        f.write(content)

def read_file(node):
    with open(node, "r") as f:
        content = f.read().strip()
        pr_debug(f"Reading file: {node} = {content}")
        return content

def create_power_stats(time_ms, uj):
    sec = time_ms / 1e3

    power = uj / time_ms
    mj = uj / 1000
    joules = mj / 1000

    return {
        "elapsed_sec": sec,
        "elapsed_ns": time_ms * 1e6,
        "power_mean": power,
        "energy_millijoules": mj,
        "energy_joules": joules,
    }

def get_cpu_freqs(cpu):
    raw_freqs = read_file(f"{SYS_CPU}/cpu{cpu}/cpufreq/scaling_available_frequencies").split(" ")
    boost_node = f"{SYS_CPU}/cpu{cpu}/cpufreq/scaling_boost_frequencies"
    # Some devices have extra boost frequencies not in scaling_available_frequencies
    if os.path.exists(boost_node):
        raw_freqs += read_file(boost_node).split(" ")

    # Need to sort because different platforms have different orders
    freqs = sorted(set(int(freq) for freq in raw_freqs if freq))

    return freqs

def init_cpus():
    print("Frequency domains: ", end="", flush=True)
    bench_cpus = []
    for policy_dir in sorted(os.listdir(f"{SYS_CPU}/cpufreq")):
        if policy_dir.startswith("policy"):
            first_cpu = int(policy_dir[len("policy"):])
            if first_cpu in REPLACE_CPUS:
                first_cpu = REPLACE_CPUS[first_cpu]

            print(f"cpu{first_cpu}", end=" ", flush=True)
            bench_cpus.append(first_cpu)
        else:
            pr_debug(f"Unrecognized file/dir in cpufreq: {policy_dir}")
            continue
    print()

    print("Offline CPUs: ", end="", flush=True)
    cpu_count = len(re.findall(r'processor\s+:\s+\d+', read_file("/proc/cpuinfo")))

    for cpu in range(cpu_count):
        if cpu == HOUSEKEEPING_CPU:
            continue

        print(f"cpu{cpu}", end=" ", flush=True)
        write_cpu(cpu, "online", "0")
    print(flush=True)

    pr_debug("Minimizing frequency of housekeeping CPU")
    min_freq = min(get_cpu_freqs(HOUSEKEEPING_CPU))
    pr_debug(f"Minimum frequency for {HOUSEKEEPING_CPU}: {min_freq} kHz")
    write_cpu(HOUSEKEEPING_CPU, "cpufreq/scaling_governor", "userspace")
    write_cpu(HOUSEKEEPING_CPU, "cpufreq/scaling_setspeed", str(min_freq))
    pr_debug()

    return bench_cpus, cpu_count

def check_charging(node, charging_value, charging_warned):
    if os.path.exists(node):
        psy_status = read_file(node)
        pr_debug(f"Power supply status at {node}: {psy_status}")
        if psy_status == charging_value and not charging_warned:
            print()
            print("=============== WARNING ===============")
            print("Detected power supply in charging state!")
            print("Power measurements will be invalid and benchmark results may be affected.")
            print("Unplug the device and restart the benchmark for valid results.")
            print("=============== WARNING ===============")
            print()
            return True

    return charging_warned

def main():
    bench_start_time = time.time()

    print(BANNER)
    pr_debug("Running in debug mode")

    pr_debug("Initializing CPU states")
    bench_cpus, cpu_count = init_cpus()

    pr_debug("Waiting for power usage to settle")
    time.sleep(30)

    pr_debug("Starting benchmark")
    pr_debug()

    cpus_data = {}
    # Tensor G4 has 4 little CPUs, 3 big CPUs, and 1 prime CPU
    cpu_to_cluster = [0, 0, 0, 0, 1, 1, 1, 2]
    # To move housekeeping over to the next cluster for better measurements
    cluster_to_affinity = [4, 7, 0]
    for cpu in bench_cpus:
        print()
        print(f"===== CPU {cpu} =====")

        cpu_data = {
            "freqs": {}
        }
        cpus_data[cpu] = cpu_data

        pr_debug("Onlining CPU")
        write_cpu(cpu, "online", "1")

        # Move housekeeping to the next cluster so power readings aren't affected
        cluster = cpu_to_cluster[cpu]
        new_housekeeping_cpu = cluster_to_affinity[cluster]
        write_cpu(new_housekeeping_cpu, "online", "1")
        write_cpu(new_housekeeping_cpu, "cpufreq/scaling_governor", "userspace")
        freqs = get_cpu_freqs(new_housekeeping_cpu)
        write_cpu(new_housekeeping_cpu, "cpufreq/scaling_setspeed", str(min(freqs)))
        os.sched_setaffinity(0, {new_housekeeping_cpu})

        pr_debug("Setting governor")
        write_cpu(cpu, "cpufreq/scaling_governor", "userspace")

        pr_debug("Getting frequencies")
        freqs = get_cpu_freqs(cpu)
        print("Frequencies:", " ".join(str(int(freq / 1000)) for freq in freqs))
        print()

        # Some kernels may change the defaults
        pr_debug("Setting frequency limits")
        write_cpu(cpu, "cpufreq/scaling_min_freq", str(min(freqs)))
        write_cpu(cpu, "cpufreq/scaling_max_freq", str(max(freqs)))
        # Sometimes, reading back the limits immediately may give an incorrect result
        pr_debug("Waiting for frequency limits to take effect")
        time.sleep(1)

        # Bail out if the kernel is clamping our values
        pr_debug("Validating frequency limits")
        real_min_freq = int(read_file(f"{SYS_CPU}/cpu{cpu}/cpufreq/scaling_min_freq"))
        if real_min_freq != min(freqs):
            raise ValueError(f"Minimum frequency setting {min(freqs)} rejected by kernel; got {real_min_freq}")
        real_max_freq = int(read_file(f"{SYS_CPU}/cpu{cpu}/cpufreq/scaling_max_freq"))
        if real_max_freq != max(freqs):
            raise ValueError(f"Maximum frequency setting {max(freqs)} rejected by kernel; got {real_max_freq}")

        for freq in freqs:
            mhz = freq / 1000
            print(f"{int(mhz):4d}: ", end="", flush=True)
            write_cpu(cpu, "cpufreq/scaling_setspeed", str(freq))

            pr_debug("Waiting for frequency to settle")
            time.sleep(0.1)

            pr_debug("Validating frequency")
            real_freq = int(read_file(f"{SYS_CPU}/cpu{cpu}/cpufreq/scaling_cur_freq"))
            if real_freq != freq:
                raise ValueError(f"Frequency setting is {freq} but kernel is using {real_freq}")

            pr_debug("Waiting for power usage to settle")
            time.sleep(3)

            pr_debug("Measuring idle power usage")
            start_ms, start_uj = sample_energy(cluster)
            time.sleep(FREQ_IDLE_TIME)
            end_ms, end_uj = sample_energy(cluster)
            idle_uj = end_uj - start_uj
            idle_ms = end_ms - start_ms
            idle_power = idle_uj / idle_ms
            idle_joules = idle_uj / 1e6
            pr_debug(f"Idle: {idle_power:4.0f} mW    {idle_joules:4.1f} J")

            pr_debug("Running CoreMark...")
            start_ms, start_uj = sample_energy(cluster)
            start_time = time.time_ns()
            cm_out = run_cmd(["taskset", "-c", f"{cpu}", "coremark", *COREMARK_PERFORMANCE_RUN])
            end_time = time.time_ns()
            end_ms, end_uj = sample_energy(cluster)
            uj = end_uj - start_uj
            ms = end_ms - start_ms

            pr_debug(cm_out)
            elapsed_sec = (end_time - start_time) / 1e9

            # Extract score and iterations
            match = re.search(r'CoreMark 1\.0 : ([0-9.]+?) / ', cm_out)
            if not match.group(1):
                if "Must execute for at least 10 secs" in cm_out:
                    raise ValueError("Benchmark ran too fast; increase COREMARK_ITERATIONS and try again")
                else:
                    print(cm_out, file=sys.stderr)
                    raise ValueError("Failed to parse CoreMark output")

            score = float(match.group(1))
            match = re.search(r'Iterations\s+:\s+(\d+)', cm_out)
            iters = float(match.group(1))

            # Calculate power values
            power = uj / ms
            # CoreMarks/MHz as per EEMBC specs
            cm_mhz = score / mhz
            # mW * sec = mJ
            mj = uj / 1000
            joules = mj / 1000
            # ULPMark-CM score = iterations per millijoule
            ulpmark_score = iters / mj

            print(f"{score:5.0f}     {cm_mhz:3.1f} C/MHz   {power:4.0f} mW   {joules:4.1f} J   {ulpmark_score:4.1f} I/mJ   {elapsed_sec:5.1f} s")

            cpu_data["freqs"][freq] = {
                "active": {
                    **create_power_stats(ms, uj),
                    "coremark_score": score,
                    "coremarks_per_mhz": cm_mhz,
                    "ulpmark_cm_score": ulpmark_score
                },
                "idle": create_power_stats(idle_ms, idle_uj),
            }

        # In case the CPU shares a freq domain with the housekeeping CPU, e.g. cpu1
        pr_debug(f"Minimizing frequency of CPU: {min(freqs)} kHz")
        write_cpu(cpu, "cpufreq/scaling_setspeed", str(min(freqs)))

        pr_debug("Offlining CPU")
        write_cpu(cpu, "online", "0")
        print()

    # Make the rest run faster
    pr_debug("Maxing housekeeping CPU frequency")
    max_hk_freq = max(get_cpu_freqs(HOUSEKEEPING_CPU))
    write_cpu(HOUSEKEEPING_CPU, "cpufreq/scaling_setspeed", str(max_hk_freq))

    # OK to GC beyond this point as all the benchmarking is done
    pr_debug("Enabling Python GC")
    gc.enable()

    print()
    print("Benchmark finished!")

    bench_finish_time = time.time()

    pr_debug("Writing JSON data")
    data = {
        "version": 1,
        "total_elapsed_sec": bench_finish_time - bench_start_time,
        "cpus": cpus_data,
        "meta": {
            "cpu_count": cpu_count,
        },
    }

    pr_debug("Writing JSON results")
    results_json = json.dumps(data)
    pr_debug(results_json)
    with open("/tmp/results.json", "w+") as f:
        f.write(results_json)

    pr_debug("Writing CSV results")
    with open("/tmp/results.csv", "w+") as f:
        fields = [
            "CPU",
            "Frequency (kHz)",
            "CoreMarks (iter/s)",
            "CoreMarks/MHz",
            "Power (mW)",
            "Energy (J)",
            "ULPMark-CM (iter/mJ)",
            "Time (s)"
        ]

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for cpu, cpu_data in cpus_data.items():
            for freq, freq_data in cpu_data["freqs"].items():
                freq_data = freq_data["active"]

                writer.writerow({
                    "CPU": cpu,
                    "Frequency (kHz)": freq,
                    "CoreMarks (iter/s)": freq_data["coremark_score"],
                    "CoreMarks/MHz": freq_data["coremarks_per_mhz"],
                    "Power (mW)": freq_data["power_mean"],
                    "Energy (J)": freq_data["energy_joules"],
                    "ULPMark-CM (iter/mJ)": freq_data["ulpmark_cm_score"],
                    "Time (s)": freq_data["elapsed_sec"],
                })

if __name__ == "__main__":
    main()
