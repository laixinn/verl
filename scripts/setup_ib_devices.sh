#!/bin/bash

IB_BASE_DIR="/sys/class/infiniband"

setup_ib_devices() {
    declare -A ib_devices

    find_ib_devices

    if [[ ${#ib_devices[@]} -eq 0 ]]; then
        echo "No IB device found."
        return 0
    fi

    filter_ib_devices

    if [[ ${#ib_devices[@]} -eq 0 ]]; then
        echo "No IB device with active RoCE v2 netdev mapping and local GPU connectivity found."
        return 0
    fi

    declare -a max_rate_devices min_rate_devices
    group_ib_devices

    declare -a sorted_max_rate_devices sorted_min_rate_devices
    sort_ib_devices

    print_ib_devices

    local gid_index
    gid_index=$(find_gid_index)

    local device_list device_mapping socket_ifname
    device_list=$(generate_device_list)
    device_mapping=$(generate_device_mapping)
    socket_ifname=$(get_socket_ifname)

    setup_envs "$gid_index" "$device_list" "$device_mapping" "$socket_ifname"
}

find_ib_devices() {
    for device in "$IB_BASE_DIR"/*; do
        [[ ! -d "$device" ]] && continue

        local device_name ports_path
        device_name=$(basename "$device")
        ports_path="$device/ports"

        [[ ! -d "$ports_path" ]] && continue

        for port in "$ports_path"/*; do
            [[ ! -d "$port" ]] && continue

            local rate_file rate_content rate_value
            rate_file="$port/rate"
            [[ ! -f "$rate_file" ]] && continue

            rate_content=$(<"$rate_file")
            rate_value=$(echo "$rate_content" | awk '{print $1}' | grep -oE '[0-9]+')

            if [[ -n "$rate_value" ]]; then
                ib_devices["$device_name"]=$rate_value
                break
            fi
        done
    done
}

read_sysfs() {
    local file="$1"
    [[ -r "$file" ]] && cat "$file" 2>/dev/null || true
}

is_usable_roce_v2_gid() {
    local port_dir="$1" gid_index="$2"
    local state link_layer gid gid_type ndev carrier

    state=$(read_sysfs "$port_dir/state")
    [[ "$state" == *"ACTIVE"* ]] || return 1

    link_layer=$(read_sysfs "$port_dir/link_layer")
    [[ "$link_layer" == "Ethernet" ]] || return 1

    gid=$(read_sysfs "$port_dir/gids/$gid_index")
    [[ -n "$gid" ]] || return 1
    [[ "$gid" != "0000:0000:0000:0000:0000:0000:0000:0000" ]] || return 1
    [[ "$gid" != "fe80:0000:0000:0000:0000:0000:0000:0000" ]] || return 1
    [[ "${gid%%:*}" == "0000" ]] || return 1

    gid_type=$(read_sysfs "$port_dir/gid_attrs/types/$gid_index")
    [[ "$gid_type" =~ [vV]2 ]] || return 1

    ndev=$(read_sysfs "$port_dir/gid_attrs/ndevs/$gid_index")
    [[ -n "$ndev" && -d "/sys/class/net/$ndev" ]] || return 1

    carrier=$(read_sysfs "/sys/class/net/$ndev/carrier")
    [[ "$carrier" == "1" ]] || return 1

    return 0
}

list_usable_gid_indices() {
    local dev="$1" port_dir gid_file gid_index

    for port_dir in "$IB_BASE_DIR/$dev/ports/"*; do
        [[ -d "$port_dir" ]] || continue

        for gid_file in "$port_dir/gids/"*; do
            [[ -f "$gid_file" ]] || continue
            gid_index=$(basename "$gid_file")

            if is_usable_roce_v2_gid "$port_dir" "$gid_index"; then
                echo "$gid_index"
            fi
        done
    done | sort -n -u
}

device_has_usable_gid_index() {
    local dev="$1" gid_index="$2" port_dir

    for port_dir in "$IB_BASE_DIR/$dev/ports/"*; do
        [[ -d "$port_dir" ]] || continue
        is_usable_roce_v2_gid "$port_dir" "$gid_index" && return 0
    done

    return 1
}

has_usable_roce_netdev() {
    local dev="$1"
    [[ -n "$(list_usable_gid_indices "$dev" | head -n 1)" ]]
}

topo_labels_for_ib() {
    local dev="$1" topo_output="$2"

    echo "$dev"
    awk -v dev="$dev" '
        $1 ~ /^NIC[0-9]+:/ {
            label = $1
            sub(/:$/, "", label)
            for (i = 2; i <= NF; i++) {
                if ($i == dev) {
                    print label
                }
            }
        }
    ' <<< "$topo_output"
}

current_gpu_topo_rows() {
    local topo_output="$1" cuda_visible="${CUDA_VISIBLE_DEVICES:-}" id any_numeric=0

    if [[ -n "$cuda_visible" && "$cuda_visible" != "all" && "$cuda_visible" != "none" && "$cuda_visible" != "void" ]]; then
        IFS=',' read -ra visible_ids <<< "$cuda_visible"
        for id in "${visible_ids[@]}"; do
            id="${id// /}"
            if [[ "$id" =~ ^[0-9]+$ ]]; then
                echo "GPU$id"
                any_numeric=1
            fi
        done

        (( any_numeric )) && return
    fi

    awk '
        NF == 0 {
            next
        }
        header_seen && $1 ~ /^GPU[0-9]+$/ {
            print $1
        }
        !header_seen {
            header_seen = 1
        }
    ' <<< "$topo_output"
}

topo_label_has_gpu_connectivity() {
    local label="$1" topo_output="$2" gpu_rows="$3"

    awk -v label="$label" -v gpu_rows="$gpu_rows" '
        BEGIN {
            split(gpu_rows, rows, /[[:space:]]+/)
            for (i in rows) {
                if (rows[i] != "") {
                    wanted_gpu[rows[i]] = 1
                }
            }
        }
        NF == 0 {
            next
        }
        !header_seen {
            for (i = 1; i <= NF; i++) {
                if ($i == label) {
                    nic_col = i + 1
                }
            }
            header_seen = 1
            next
        }
        $1 ~ /^GPU[0-9]+$/ && nic_col > 0 {
            if (!($1 in wanted_gpu)) {
                next
            }

            cell = $nic_col
            if (cell == "PIX" || cell == "PXB" || cell == "PXN" ||
                cell == "PHB" || cell ~ /^NV[0-9]+$/) {
                ok = 1
            }
        }
        END {
            exit ok ? 0 : 1
        }
    ' <<< "$topo_output"
}

has_gpu_topology_connectivity() {
    local dev="$1" topo_output="$2" label
    local gpu_rows

    [[ -n "$topo_output" ]] || return 1

    gpu_rows=$(current_gpu_topo_rows "$topo_output" | xargs)
    [[ -n "$gpu_rows" ]] || return 1

    while IFS= read -r label; do
        [[ -n "$label" ]] || continue
        if topo_label_has_gpu_connectivity "$label" "$topo_output" "$gpu_rows"; then
            return 0
        fi
    done < <(topo_labels_for_ib "$dev" "$topo_output")

    return 1
}

filter_ib_devices() {
    local topo_output dev

    topo_output=$(nvidia-smi topo -m 2>/dev/null || true)

    for dev in "${!ib_devices[@]}"; do
        if ! has_usable_roce_netdev "$dev"; then
            echo "Skip $dev: no active RoCE v2 GID mapped to an up netdev."
            unset "ib_devices[$dev]"
            continue
        fi

        if ! has_gpu_topology_connectivity "$dev" "$topo_output"; then
            echo "Skip $dev: no local GPU connectivity in nvidia-smi topo -m."
            unset "ib_devices[$dev]"
            continue
        fi
    done
}

group_ib_devices() {
    local max_rate=-1 min_rate=99999

    for dev in "${!ib_devices[@]}"; do
        local rate=${ib_devices[$dev]}

        if (( rate >= max_rate )); then
            if (( rate > max_rate )); then
                max_rate=$rate
                max_rate_devices=()
            fi
            max_rate_devices+=("$dev")
        fi

        if (( rate <= min_rate )); then
            if (( rate < min_rate )); then
                min_rate=$rate
                min_rate_devices=()
            fi
            min_rate_devices+=("$dev")
        fi
    done
}

sort_ib_devices() {
    sorted_max_rate_devices=($(
            for dev in "${max_rate_devices[@]}"; do
                suffix="${dev##*_}"
                echo "$suffix $dev"
            done | sort -n | awk '{print $2}'
    ))

    sorted_min_rate_devices=($(
            for dev in "${min_rate_devices[@]}"; do
                suffix="${dev##*_}"
                echo "$suffix $dev"
            done | sort -n | awk '{print $2}'
    ))
}

find_gid_index() {
    local gid_index=7 candidate dev all_match

    [[ ${#sorted_max_rate_devices[@]} -gt 0 ]] || {
        echo "$gid_index"
        return
    }

    while IFS= read -r candidate; do
        [[ -n "$candidate" ]] || continue

        all_match=1
        for dev in "${sorted_max_rate_devices[@]}"; do
            if ! device_has_usable_gid_index "$dev" "$candidate"; then
                all_match=0
                break
            fi
        done

        if (( all_match )); then
            echo "$candidate"
            return
        fi
    done < <(list_usable_gid_indices "${sorted_max_rate_devices[0]}")

    echo "$gid_index"
}

generate_device_list() {
    local IFS=','
    echo "${sorted_max_rate_devices[*]}"
}

generate_device_mapping() {
    local mappings=()
    for dev in "${sorted_max_rate_devices[@]}"; do
        mappings+=("$dev:1")
    done
    local IFS=','
    echo "${mappings[*]}"
}

ib_to_net() {
    local ib_device="$1"
    local net_device="eth0"

    local script_dir="$(dirname "$(readlink -f "$0")")"
    local result=$("$script_dir/ibdev2netdev.sh" 2>/dev/null | grep "^${ib_device} port")

    if [ -z "$result" ]; then
        echo "$net_device"
        return 1
    fi

    net_device=$(echo "$result" | sed 's/.*==> \([^ ]*\) .*/\1/')

    echo "$net_device"
}

get_socket_ifname() {
    local ib_idx=$(( ${#sorted_min_rate_devices[@]} - 1 ))
    local ib_device="${sorted_min_rate_devices[$ib_idx]}"
    local net_device=$(ib_to_net "$ib_device")

    echo "$net_device"
}

print_ib_devices() {
    local devices=()
    for dev in "${!ib_devices[@]}"; do
        devices+=("($dev, ${ib_devices[$dev]})")
    done
    local IFS=,
    printf "All IB devices(device_name, rate): %s\n" "${devices[*]}"
    printf "Max rate IB devices: %s\n" "${sorted_max_rate_devices[*]}"
    printf "Min rate IB devices: %s\n" "${sorted_min_rate_devices[*]}"
}

setup_envs() {
    local gid_index=$1 device_list=$2 device_mapping=$3 socket_ifname=$4

    # Export into the current shell first so callers that merely `source`
    # (or run) this script can consume the variables immediately, even when
    # writing the persistent ENV_FILE below is not possible (e.g. no
    # permission to write under /etc/profile.d in unprivileged/CI shells).

    # NCCL
    export NCCL_IB_GID_INDEX=$gid_index
    export NCCL_IB_HCA=$device_list
    export NCCL_SOCKET_IFNAME=$socket_ifname

    # MSCCL
    export MSCCLPP_IB_GID_INDEX=$gid_index
    export MSCCLPP_HCA_DEVICES=$device_list
    export MSCCLPP_SOCKET_IFNAME=$socket_ifname

    # NVSHMEM
    export NVSHMEM_IB_GID_INDEX=$gid_index
    export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
    export NVSHMEM_HCA_PE_MAPPING=$device_mapping
    export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=$socket_ifname

    # MoonCake
    export MC_GID_INDEX=$gid_index
    export MC_IB_DEVICES=$device_list
}

setup_ib_devices

