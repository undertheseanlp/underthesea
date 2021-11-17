import logging.config

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"standard": {"format": "%(asctime)-15s %(message)s"}},
        "handlers": {
            "console": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "underthesea": {"handlers": ["console"], "level": "INFO", "propagate": False}
        },
    }
)

logger = logging.getLogger("underthesea")


###########################################################
# Print Table with orgtbl (replace tabulate)
###########################################################
def generate_table(data, headers=None):
    content = ''
    # calculate max_widths
    num_columns = len(data[0])
    max_widths = [0] * len(data[0])
    for item in data:
        for i in range(num_columns):
            max_widths[i] = max(max_widths[i], len(str(item[i])))
    # generate table headers
    if headers:
        for i, item in enumerate(headers):
            max_widths[i] = max(max_widths[i], len(str(item)))
        header_str = ''
        for i, item in enumerate(headers):
            header_str += '| {item:{width}} '.format(item=item, width=max_widths[i])
        header_str += '|\n'
        for i, item in enumerate(headers):
            border_str = '|' if i == 0 else '+'
            header_str += border_str + '-' * (max_widths[i] + 2)
        header_str += '|\n'
    content += header_str
    # generate table body
    for row in data:
        data_str = ''
        for i, item in enumerate(row):
            data_str += '| {item:{width}} '.format(item=str(item), width=max_widths[i])
        data_str += '|\n'
        content += data_str
    return content


def print_table(data, headers=None):
    content = generate_table(data, headers)
    print(content)
