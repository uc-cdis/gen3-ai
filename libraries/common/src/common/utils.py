from logging import Logger


def log_api_call(service: str, logging: Logger, debug_log: str | None = None, **kwargs):
    """
    Logs a INFO level response from the {service} in a standard format with the
    provided kwargs as CSV.

    Args:
        logging (Logger): the logger to use, must be provided for the context of the file that is actually logging.
            if we instantiated here, the log would look like it's coming from the utils file directly.
        debug_log (str): Optional debug log message
        **kwargs: Additional keyword arguments to include in the log message
    """
    log_message = f"{service} API Call. "
    for kwarg, value in kwargs.items():
        log_message += f"{kwarg}={value}, "
    log_message = log_message.rstrip(", ")

    logging.info(log_message)

    if debug_log:
        logging.debug(f"{debug_log}")
