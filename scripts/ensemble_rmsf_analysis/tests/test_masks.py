from pathlib import Path
import tempfile
import unittest

from scripts.ensemble_rmsf_analysis.masks import (
    compact_ranges, extract_a3m_mask, parse_ranges,
)


class MaskTests(unittest.TestCase):
    def test_range_parser_is_inclusive(self):
        self.assertEqual(parse_ranges("1, 3-5, 8"), {1, 3, 4, 5, 8})
        self.assertEqual(compact_ranges({1, 3, 4, 5, 8}), "1, 3-5, 8")

    def test_a3m_parser_removes_lowercase_and_preserves_query_axis(self):
        text = ">query\nACD-EF\n>one\nAXdD-XF\n>two\nAXD-XX\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.a3m"
            path.write_text(text)
            table, summary = extract_a3m_mask(path, expected_query_length=5)
        self.assertEqual(summary["query_length"], 5)
        self.assertEqual(set(table.loc[table.directly_masked, "raw_residue_number"]), {2, 4})

    def test_authoritative_checkpoint_counts(self):
        kv21 = parse_ranges("288-328,370-384,401-417")
        self.assertEqual(len(kv21), 73)
        wt = parse_ranges("127,160,163,173,199,213,235-287,401,407-412,557,570,594,602,607,617-671,750,753,755,757,937,947,993,1013-1068,1181,1185,1187,1189,1242,1275,1285,1301,1307,1317,1369-1426,1519,1521-1522,1524,1527,1533")
        g402s = wt | parse_ranges("397-400,402-406")
        g406r = wt | parse_ranges("402-406")
        g490r = wt | parse_ranges("485-495")
        self.assertEqual([len(x) for x in (wt, g402s, g406r, g490r)], [263, 272, 268, 274])
        self.assertEqual(g402s - wt, parse_ranges("397-400,402-406"))
        self.assertEqual(g406r - wt, parse_ranges("402-406"))
        self.assertEqual(g490r - wt, parse_ranges("485-495"))
        self.assertTrue(402 in g402s and 406 in g406r and 490 in g490r)
