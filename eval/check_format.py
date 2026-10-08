"""Check that a results workbook has the same format as an older one: only theme columns (and the themes-table rows) may differ.

    python eval/check_format.py --old runs/old_open_ends/aspect_sentiment_results.xlsx --new runs/beverage/aspect_sentiment_results.xlsx

Asserts: same sheet names in the same order; on the question sheets the same leading columns (id, Answer, Overall) at the same
positions; cell values only Positive / Negative / Neutral / blank (0/1/2 on results_codes, blank otherwise); same header style,
freeze panes, column widths and cell colours. Prints the sheet / column name differences.
"""
import argparse
import openpyxl

WORDS = {"Positive": "C6EFCE", "Negative": "FFC7CE", "Neutral": "FFEB9C"}
FIXED = ["Answer", "Overall"]
SKIP_SHEETS = {"detail", "pairs", "themes", "timings"}          # tables, not result sheets


def style_sig(c):
    return (c.font.b, c.font.color.rgb if c.font.color else None, c.fill.fgColor.rgb if c.fill and c.fill.fill_type else None,
            c.alignment.wrap_text, c.alignment.vertical)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    a = ap.parse_args()
    old, new = openpyxl.load_workbook(a.old), openpyxl.load_workbook(a.new)
    names = lambda wb: [n for n in wb.sheetnames if n != "Combined"]          # the Combined sheet is new: not part of the old format
    assert names(old) == names(new), f"sheet names differ: {names(old)} vs {names(new)}"
    print("sheet names identical:", names(new))
    for name in names(new):
        wo, wn = old[name], new[name]
        ho, hn = [c.value for c in wo[1]], [c.value for c in wn[1]]
        gone, added = [h for h in ho if h not in hn], [h for h in hn if h not in ho]
        print(f"\n[{name}] columns {len(ho)} -> {len(hn)}")
        print("   removed:", gone or "-")
        print("   added:  ", added or "-")
        if name in SKIP_SHEETS:
            if name == "timings":
                assert ho == hn, "timings columns changed"
            continue
        for h in ["respondent_id"] + FIXED if "respondent_id" in ho else FIXED:        # same position of the fixed columns
            assert ho.index(h) == hn.index(h), f"[{name}] column {h!r} moved: {ho.index(h)} -> {hn.index(h)}"
        lead = ho.index("Overall") + 1
        assert ho[:lead] == hn[:lead], f"[{name}] leading columns differ: {ho[:lead]} vs {hn[:lead]}"
        # vocabulary
        seen = set()
        for j, h in enumerate(hn, 1):
            if h in ("respondent_id", "Answer"):
                continue
            vals = {wn.cell(r, j).value for r in range(2, wn.max_row + 1)}
            allowed = {*WORDS, None}
            assert vals <= allowed, f"[{name}] column {h!r} has unexpected values {vals - allowed}"
            seen |= vals
        print("   cell vocabulary:", sorted(v for v in seen if v))
        # styles: header, freeze panes, widths, colours, wrap
        assert wo.freeze_panes == wn.freeze_panes, "freeze panes differ"
        for h in ["Answer", "Overall"]:
            assert wo.column_dimensions[openpyxl.utils.get_column_letter(ho.index(h) + 1)].width == \
                wn.column_dimensions[openpyxl.utils.get_column_letter(hn.index(h) + 1)].width, f"width of {h} differs"
        assert style_sig(wo.cell(1, 1)) == style_sig(wn.cell(1, 1)), "header style differs"
        assert len({style_sig(c) for c in wn[1]}) == 1, "header cells are not styled alike"
        checked = 0
        for row in wn.iter_rows(min_row=2):
            for c in row:
                if c.value in WORDS:
                    assert c.fill.fgColor.rgb.endswith(WORDS[c.value]), f"[{name}] {c.coordinate} {c.value!r} has fill {c.fill.fgColor.rgb}"
                    checked += 1
                elif c.value is None and c.column > 4:
                    assert not (c.fill and c.fill.fill_type == "solid"), f"[{name}] blank cell {c.coordinate} is coloured"
        assert checked, f"[{name}] no coloured cells found"
        so = {(c.value, style_sig(c)) for row in wo.iter_rows(min_row=2) for c in row if c.value in WORDS}
        sn = {(c.value, style_sig(c)) for row in wn.iter_rows(min_row=2) for c in row if c.value in WORDS}
        assert len({v for v, _ in sn}) == len(sn), "one value has several different styles"      # same value -> same look
        both = {v for v, _ in so} & {v for v, _ in sn}
        assert {x for x in so if x[0] in both} == {x for x in sn if x[0] in both}, "cell style of a value differs from the old workbook"
        print(f"   styles identical (header, freeze {wn.freeze_panes}, widths, {checked} coloured cells checked, wrap/top alignment)")
    for name in ("results_codes",):
        if name in new.sheetnames:
            wn = new[name]
            for row in wn.iter_rows(min_row=2):
                for c in row:
                    if isinstance(c.value, (int, float)) and c.column > 2:
                        assert c.value in (0, 1, 2), f"results_codes has {c.value}"
    print("\nFORMAT CHECK PASSED")


if __name__ == "__main__":
    main()
