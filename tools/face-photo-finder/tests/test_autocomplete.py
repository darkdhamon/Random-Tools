import unittest

from face_finder.app import autocomplete_matches, identity_display_label, identity_id_from_label
from face_finder.catalog import KnownIdentity


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

    def test_duplicate_name_identity_labels_round_trip_the_stable_id(self) -> None:
        first = KnownIdentity(12, "Same Name", ())
        second = KnownIdentity(34, "Same Name", ())
        self.assertEqual(identity_display_label(first), "Same Name (#12)")
        self.assertEqual(identity_display_label(second), "Same Name (#34)")
        self.assertEqual(identity_id_from_label(identity_display_label(second)), 34)


if __name__ == "__main__":
    unittest.main()
