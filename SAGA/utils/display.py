from astropy.io import ascii

try:
    import pyperclip
except ImportError:
    _HAS_CLIP = False
else:
    _HAS_CLIP = True
try:
    from IPython.display import display_html
except ImportError:
    _HAS_DISPLAY = False
else:
    _HAS_DISPLAY = True

__all__ = ["print_for_viewer", "read_marks_from_clipboard", "show_images"]


def print_for_viewer(table, keys=("OBJID", "RA", "DEC"), additional_keys=None):
    keys = list(keys)
    if additional_keys:
        keys.extend(additional_keys)
    output = "\n".join(table[keys].pformat_all(-1, -1))
    if _HAS_CLIP:
        pyperclip.copy(output)
    print(output)


def read_marks_from_clipboard(extract_marked_value=True, extract_cols="objid", marked_col_name="marked"):
    if not _HAS_CLIP:
        raise RuntimeError("needs pyperclip to work")
    t = ascii.read(table=pyperclip.paste(), format="fast_tab")
    if extract_marked_value is True or extract_marked_value is False:
        extract_marked_value = str(extract_marked_value).lower()
    if extract_marked_value is not None:
        t = t[t[marked_col_name] == extract_marked_value]
    if extract_cols is not None:
        t = t[extract_cols]
    return t


def show_images(
    table,
    pixscale=0.2,
    size=200,
    keys=(
        "OBJID",
        "RA",
        "DEC",
        "HOSTID",
        "r_mag",
        "sb_r",
        "gr",
        "SPEC_Z",
        "ZQUALITY",
        "TELNAME",
    ),
    layer="ls-dr9",
    ra_label="RA",
    dec_label="DEC",
    use_dev=False,
    additional_keys=None,
):
    out = []
    dev = "-dev" if use_dev else ""
    keys_used = [key for key in keys if key in table.colnames] if keys else []
    if additional_keys:
        keys_used.extend(additional_keys)
    if dec_label not in keys_used:
        keys_used = [dec_label] + keys_used
    if ra_label not in keys_used:
        keys_used = [ra_label] + keys_used
    for row in table:

        url = "https://www.legacysurvey.org/viewer{dev}/jpeg-cutout/?ra={ra}&dec={dec}&pixscale={pixscale}&layer={layer}&size={size}".format(
            ra=row[ra_label],
            dec=row[dec_label],
            layer=layer,
            pixscale=pixscale,
            size=size,
            dev=dev,
        )

        link = "https://www.legacysurvey.org/viewer{dev}?ra={ra}&dec={dec}&layer={layer}&zoom=16".format(
            ra=row[ra_label],
            dec=row[dec_label],
            layer=layer,
            dev=dev,
        )
        title = "\n".join((f"{k} = {row[k]}" for k in keys_used))

        out.append(
            f'<a href="{link}" target="_blank"><img src="{url}" style="display:inline-block; width:{size}px; height:{size}px" title="{title}" /></a>'
        )

    html = "\n".join(out)

    if _HAS_DISPLAY:
        display_html(html, raw=True)
        return

    return html
