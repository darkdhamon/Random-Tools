import unittest

from face_finder.app import autocomplete_matches


class AutocompleteTests(unittest.TestCase):
    def test_prefixes_are_case_insensitive_and_ranked_first(self) -> None:
        names = ["Alice Smith", "Malcolm Alice", "Alex Jones", "Bob"]
        self.assertEqual(
            autocomplete_matches("al", names),
            ["Alice Smith", "Alex Jones", "Malcolm Alice"],
        )

    def test_empty_query_restores_all_names(self) -> None:
        names = ["Alice", "Bob"]
        self.assertEqual(autocomplete_matches("", names), names)

    def test_unmatched_text_allows_a_new_name(self) -> None:
        self.assertEqual(autocomplete_matches("New Person", ["Alice", "Bob"]), [])


if __name__ == "__main__":
    unittest.main()
