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
print_line() {
  local command=$1
  local message_left=$2
  local service=$3
  local message_right=$4

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
  printf '%b' "${PURPLE}"
  printf '%*s' "$left_dashes" '' | tr ' ' '-'
  printf '%b' " ${BLUE}$command ${PURPLE}$message_left ${CYAN}$service ${PURPLE}$message_right "
  printf '%*s\n' "$message_right_dashes" '' | tr ' ' '-'
  printf '%b' "${RESET}"
}


report_error() {
  local status=$1
  local msg=$2
  if [ $status -ne 0 ]; then
    echo
    echo "${RED}** ERROR: $msg **${RESET}"
  fi
}
