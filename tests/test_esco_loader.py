import unittest

from melo_benchmark.data_processing.esco_loader import EscoLoader


# noinspection DuplicatedCode
class TestEscoLoader(unittest.TestCase):

    def test_load_1_0_3(self):
        version = "1.0.3"
        esco_loader = EscoLoader()
        concepts = esco_loader.load(version)

        c_number = "ab7bccb2-6f81-4a3d-a0c0-fca5d47d2775"
        c_id = f"http://data.europa.eu/esco/occupation/{c_number}"
        retrieved_concept = concepts[c_id]

        actual = retrieved_concept["id"]
        expected = c_id
        self.assertEqual(actual, expected, "Wrong concept ID.")

        actual = retrieved_concept["std_name"]["hu"]
        expected = "ipari tervező"
        self.assertEqual(actual, expected, "Wrong std name in Hungarian.")

        actual = len(retrieved_concept["alt_names"]["ro"])
        expected = 9
        self.assertEqual(actual, expected, "Wrong alt name in Romanian.")

        actual = list(retrieved_concept["description"].keys())
        expected = ["en"]
        self.assertEqual(actual, expected, "Wrong list of descriptions.")

    def test_load_1_0_8(self):
        version = "1.0.8"
        esco_loader = EscoLoader()
        concepts = esco_loader.load(version)

        c_number = "ab7bccb2-6f81-4a3d-a0c0-fca5d47d2775"
        c_id = f"http://data.europa.eu/esco/occupation/{c_number}"
        retrieved_concept = concepts[c_id]

        actual = retrieved_concept["id"]
        expected = c_id
        self.assertEqual(actual, expected, "Wrong concept ID.")

        actual = retrieved_concept["std_name"]["hu"]
        expected = "ipari tervező"
        self.assertEqual(actual, expected, "Wrong std name in Hungarian.")

        actual = len(retrieved_concept["alt_names"]["ro"])
        expected = 9
        self.assertEqual(actual, expected, "Wrong alt name in Romanian.")

        actual = len(retrieved_concept["description"].keys())
        expected = 24
        self.assertEqual(actual, expected, "Wrong list of descriptions.")

    def test_load_1_0_9(self):
        version = "1.0.9"
        esco_loader = EscoLoader()
        concepts = esco_loader.load(version)

        c_number = "ab7bccb2-6f81-4a3d-a0c0-fca5d47d2775"
        c_id = f"http://data.europa.eu/esco/occupation/{c_number}"
        retrieved_concept = concepts[c_id]

        actual = retrieved_concept["id"]
        expected = c_id
        self.assertEqual(actual, expected, "Wrong concept ID.")

        actual = retrieved_concept["std_name"]["hu"]
        expected = "ipari tervező"
        self.assertEqual(actual, expected, "Wrong std name in Hungarian.")

        actual = len(retrieved_concept["alt_names"]["ro"])
        expected = 9
        self.assertEqual(actual, expected, "Wrong alt names in Romanian.")

        actual = len(retrieved_concept["description"].keys())
        expected = 24
        self.assertEqual(actual, expected, "Wrong list of descriptions.")

    def test_load_1_1_0(self):
        version = "1.1.0"
        esco_loader = EscoLoader()
        concepts = esco_loader.load(version)

        c_number = "ab7bccb2-6f81-4a3d-a0c0-fca5d47d2775"
        c_id = f"http://data.europa.eu/esco/occupation/{c_number}"
        retrieved_concept = concepts[c_id]

        actual = retrieved_concept["id"]
        expected = c_id
        self.assertEqual(actual, expected, "Wrong concept ID.")

        actual = retrieved_concept["std_name"]["hu"]
        expected = "ipari tervező"
        self.assertEqual(actual, expected, "Wrong std name in Hungarian.")

        actual = len(retrieved_concept["alt_names"]["ro"])
        expected = 9
        self.assertEqual(actual, expected, "Wrong alt names in Romanian.")

        actual = len(retrieved_concept["description"].keys())
        expected = 24
        self.assertEqual(actual, expected, "Wrong list of descriptions.")

    def test_load_1_1_1(self):
        version = "1.1.1"
        esco_loader = EscoLoader()
        concepts = esco_loader.load(version)

        c_number = "ab7bccb2-6f81-4a3d-a0c0-fca5d47d2775"
        c_id = f"http://data.europa.eu/esco/occupation/{c_number}"
        retrieved_concept = concepts[c_id]

        actual = retrieved_concept["id"]
        expected = c_id
        self.assertEqual(actual, expected, "Wrong concept ID.")

        actual = retrieved_concept["std_name"]["hu"]
        expected = "ipari tervező"
        self.assertEqual(actual, expected, "Wrong std name in Hungarian.")

        actual = len(retrieved_concept["alt_names"]["ro"])
        expected = 24
        self.assertEqual(actual, expected, "Wrong alt names in Romanian.")

        actual = len(retrieved_concept["description"].keys())
        expected = 24
        self.assertEqual(actual, expected, "Wrong list of descriptions.")

    def test_load_1_2_0(self):
        version = "1.2.0"
        esco_loader = EscoLoader()
        concepts = esco_loader.load(version)

        c_number = "ab7bccb2-6f81-4a3d-a0c0-fca5d47d2775"
        c_id = f"http://data.europa.eu/esco/occupation/{c_number}"
        retrieved_concept = concepts[c_id]

        actual = retrieved_concept["id"]
        expected = c_id
        self.assertEqual(actual, expected, "Wrong concept ID.")

        actual = retrieved_concept["std_name"]["hu"]
        expected = "ipari tervező"
        self.assertEqual(actual, expected, "Wrong std name in Hungarian.")

        actual = len(retrieved_concept["alt_names"]["ro"])
        expected = 24
        self.assertEqual(actual, expected, "Wrong alt names in Romanian.")

        actual = len(retrieved_concept["description"].keys())
        expected = 24
        self.assertEqual(actual, expected, "Wrong list of descriptions.")
