import difflib
import re
import svgutils
from IPython.display import SVG, display, Markdown
from .utils import *

DIAGRAM_DEFAULT_IGNORE = ("system", "macro_in_system", "1bit_x_1bit_mac")


def grab_from_yaml_file(
    yaml_file, startfrom=None, same_indent=True, include_lines_before=0
):
    with open(yaml_file, "r") as f:
        contents = f.readlines()
    start, end = 0, len(contents)
    n_whitespace = 0
    if startfrom is None:
        return "".join(contents)
    for i, line in enumerate(contents):
        if re.findall(r"\b\s*" + startfrom + r"\b", line):
            start = i
            n_whitespace = len(re.findall(r"^\s*", line)[0])
            break
    else:
        raise ValueError(f"{startfrom} not found in {yaml_file}")
    for i, line in enumerate(contents[start + 1 :]):
        ws = len(re.findall(r"^\s*", line)[0])
        if ws < n_whitespace or (not same_indent and ws == n_whitespace):
            end = start + i + 1
            break
    return "".join(
        c[n_whitespace:] for c in contents[start - include_lines_before : end]
    )


def scale_svg(svg, scale=0.5):
    svg = svgutils.transform.fromstring(svg.decode("ascii"))
    svg = svgutils.compose.Figure(svg.width, svg.height, svg.getroot())
    svg = svg.scale(scale)
    svg.width = svg.width * scale
    svg.height = svg.height * scale
    return svg


def display_diagram(diagram, scale=0.5):
    display(SVG(scale_svg(diagram.create_svg(), scale).tostr()))


def display_markdown(markdown):
    display(Markdown(markdown))


def display_yaml_file(*args, **kwargs):
    display_yaml_str(grab_from_yaml_file(*args, **kwargs))


def display_yaml_str(yaml_str):
    display_markdown(f"```yaml\n{yaml_str}```")


def get_yaml_file_markdown(yaml_file, *args, **kwargs):
    return f"```yaml\n{grab_from_yaml_file(yaml_file, *args, **kwargs)}```"


def get_yaml_str_markdown(yaml_str):
    return f"```yaml\n{yaml_str}```"


def get_important_variables_markdown(name: str):
    result = []
    result.append(f"Some of the important variables for {name}:\n")

    def pfmat(key, value, note=""):
        result.append(f"- *{key}*: {value} {note if note else ''}")

    s = get_spec(name)._process()
    for v in [
        ("ARRAY_WORDLINES", "rows in the array"),
        ("ARRAY_BITLINES", "columns in the array"),
        (
            "ARRAY_PARALLEL_INPUTS",
            "input slice(s) consumed in each cycle.",
        ),
        (
            "ARRAY_PARALLEL_WEIGHTS",
            "weights slice(s) used for computation in each cycle.",
        ),
        ("ARRAY_PARALLEL_OUTPUTS", "output(s) produced in each cycle."),
        ("TECHNOLOGY", "nm"),
        ("ADC_RESOLUTION", "bit(s)"),
        ("DAC_RESOLUTION", "bit(s)"),
        ("N_ADC_PER_BANK", "ADC(s)"),
        ("SUPPORTED_INPUT_BITS", "bit(s)"),
        ("SUPPORTED_OUTPUT_BITS", "bit(s)"),
        ("SUPPORTED_WEIGHT_BITS", "bit(s)"),
        ("BITS_PER_CELL", "bit(s)"),
        (
            "CIM_UNIT_WIDTH_CELLS",
            "adjacent cell(s) in a wordline store bit(s) in one weight slice and process one input & output slice together",
        ),
        (
            "CIM_UNIT_DEPTH_CELLS",
            "adjacent cell(s) in a bitline operate in separate cycles",
        ),
        "CELL_CONFIG",
        ("GLOBAL_CYCLE_SECONDS", "clock period"),
    ]:
        if isinstance(v, tuple):
            pfmat(v[0], s.variables.get(v[0], None), v[1])
        else:
            pfmat(v, s.variables.get(v[0], None))

    return "\n".join(result)


def clean_old_output_files(max_files=50):
    out_path = os.path.join(THIS_SCRIPT_DIR, "..", "outputs")
    files = sorted(
        list(os.path.join(out_path, f) for f in os.listdir(out_path)),
        key=lambda x: os.path.getmtime(x),
    )
    while len(files) > max_files:
        shutil.rmtree(
            files.pop(0),
            ignore_errors=True,
        )


def run_test(
    macro_name: str,
    test_name: str,
    show_doc: bool = True,
    *args,
    **kwargs,
):
    test_func = get_test(macro_name, test_name)
    if show_doc:
        doc = test_func.__doc__
        doc = "\n".join([line[1:] for line in doc.split("\n")])
        display_markdown(doc)
    t = test_func(*args, **kwargs)
    clean_old_output_files()
    return t


def diff_str(a, b):
    new_a, new_b = [], []
    a = re.findall(r"[\w\.]+|\s+|.", a)
    b = re.findall(r"[\w\.]+|\s+|.", b)
    # print(f'Diffing {a} and {b}')
    matcher = difflib.SequenceMatcher(None, a, b)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            new_a.extend(a[i1:i2])
            new_b.extend(b[j1:j2])
        elif tag == "replace":
            new_a.extend([f"\033[31m{l}\033[0m" for l in a[i1:i2]])
            new_b.extend([f"\033[31m{l}\033[0m" for l in b[j1:j2]])
        elif tag == "delete":
            new_a.extend([f"\033[31m{l}\033[0m" for l in a[i1:i2]])
        elif tag == "insert":
            new_b.extend([f"\033[31m{l}\033[0m" for l in b[j1:j2]])
    return "".join(new_a), "".join(new_b)


def print_side_by_side(a, b):
    a_lines = a.splitlines()
    b_lines = b.splitlines()

    # Use difflib to match up lines
    matcher = difflib.SequenceMatcher(None, a_lines, b_lines)
    # Insert blank lines to line up the matches
    a = []
    b = []
    for _, i1, i2, j1, j2 in matcher.get_opcodes():
        a.extend(a_lines[i1:i2])
        b.extend(b_lines[j1:j2])
        a.extend([""] * (len(b) - len(a)))
        b.extend([""] * (len(a) - len(b)))

    max_a_len = max(len(line) for line in a)
    a = [line.ljust(max_a_len) for line in a]

    for i in range(len(a)):
        a[i], b[i] = diff_str(a[i], b[i])
        if a[i] and not b[i]:
            a[i] = f"\033[31m{a[i]}\033[0m"
        elif not a[i] and b[i]:
            b[i] = f"\033[31m{b[i]}\033[0m"

    for a_line, b_line in zip(a, b):
        print(f"{a_line}   |   {b_line}")
