# setup some color to help differentiate between services and tool runs
# should default to normal colors if terminal doesn't support
is_tty() { [[ -t 1 ]] && [[ -t 0 ]]; }
if is_tty; then
    RED=$(tput setaf 1)
    GREEN=$(tput setaf 2)
    YELLOW=$(tput setaf 3)
    BLUE=$(tput setaf 4)
    PURPLE=$(tput setaf 5)
    CYAN=$(tput setaf 6)
    RESET=$(tput sgr0)
else
    RED='' GREEN='' YELLOW='' BLUE='' PURPLE='' CYAN='' RESET=''
fi

# utility for adding color and a border to a message
print_header() {
  local command=$1
  local message_left=$2
  local service=$3
  local message_right=$4
  local type=${5:-default}

  local color
  case "$type" in
    error)   color="${RED}" ;;
    success) color="${GREEN}" ;;
    # anything else (default)
    *)       color="${PURPLE}" ;;
  # I hate bash syntax. esac...
  esac

  # fallback width
  local cols=$(tput cols 2>/dev/null || echo 100)
  if (( cols <= 0 )); then cols=100; fi

  local visible_length=$(
    echo -ne "$command$message_left$service$message_right" |
    sed -r 's/\x1b\[[0-9;]*m//g' |
    wc -c)

  # dashes on each side
  local left_dashes=$(( (cols - visible_length) / 2 ))
  local message_right_dashes=$(( cols - visible_length - left_dashes ))

  # print the dashes, the coloured parts, then the message_right-side dashes
  printf '%b' "${color}"
  printf '%*s' "$left_dashes" '' | tr ' ' '-'
  printf '%b' " ${BLUE}$command ${color}$message_left ${CYAN}$service ${color}$message_right "
  printf '%*s\n' "$message_right_dashes" '' | tr ' ' '-'
  printf '%b' "${RESET}"
}


print_message() {
  local message=$1
  local type=${2:-info}

  # Choose color based on type
  local color
  case "$type" in
    error) color="${RED}" ;;
    *)     color="${GREEN}" ;;  # anything else (info, success, etc.)
  esac

  # fallback width
  local cols=$(tput cols 2>/dev/null || echo 100)
  if (( cols <= 0 )); then cols=100; fi

  local visible_length=$(
    echo -ne "$message" |
    sed -r 's/\x1b\[[0-9;]*m//g' |
    wc -c
  )
  local left_spaces=$(( (cols - visible_length) / 2 ))
  local right_spaces=$(( cols - visible_length - left_spaces ))

  printf '%b' "${color}"
  printf '%*s' "$left_spaces" '' | tr ' ' ' '
  printf '%b' " ** $message ** "
  printf '%*s\n' "$right_spaces" '' | tr ' ' ' '
  printf '%b' "${RESET}"
  echo
}

report_error_or_success() {
  local status=$1
  local command=$2
  local message_left=$3
  local service=$4
  local message_right=$5

  echo
  if [ $status -ne 0 ]; then
    print_header "$command" "ERROR: $message_left" "$service" "$message_right" "error"
  else
    print_header "$command" "SUCCESS: $message_left" "$service" "$message_right" "success"
  fi
  echo
}

report_error_if_failed() {
  local status=$1
  local command=$2
  local message_left=$3
  local service=$4
  local message_right=$5

  if [ $status -ne 0 ]; then
    echo
    print_header "$command" "ERROR: $message_left" "$service" "$message_right" "error"
    echo
  fi
}

# remove any VIRTUAL_ENV to remove uv warnings about envs that aren't this one
unset VIRTUAL_ENV
