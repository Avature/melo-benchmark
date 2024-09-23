import unittest

from melo_benchmark.data_processing.crosswalk_loader import CrosswalkLoader


# noinspection DuplicatedCode
class TestCrosswalkLoader(unittest.TestCase):

    def test_load_usa_en(self):
        crosswalk_name = "usa_en"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = "11-1011.00"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Chief Executives"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"][:32]
        expected = "Determine and formulate policies"
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = "53-7121.00"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Tank Car, Truck, and Ship Loaders"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"][:41]
        expected = "Load and unload chemicals and bulk solids"
        self.assertEqual(actual, expected, "Wrong description.")

    def test_load_aut_de(self):
        crosswalk_name = "aut_de"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = "730105"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Offizier/in (Bundesheer)"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = "898102"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Zauber(er)in"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

    def test_load_bel_fr(self):
        crosswalk_name = "bel_fr"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = "OP-410"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Responsable production et technique en spectacle et " \
                   "l'audiovisuelle (h/f/x)"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = "OP-718"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Digital sales assistant (h/f/x)"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

    def test_load_bel_nl(self):
        crosswalk_name = "bel_nl"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = "OP-410"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Productieverantwoordelijke podium/audiovisueel"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = "OP-718"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Digital sales assistant"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

    def test_load_bgr_bg(self):
        crosswalk_name = "bgr_bg"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = "http://data.europa.eu/esco/isco/C8160"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Машинни оператори в хранително-вкусовата промишленост"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = "http://data.europa.eu/esco/concept/bulgarian-occupations/" \
               "52232001"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Продавач-консултант"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

    def test_load_cze_cs(self):
        crosswalk_name = "cze_cs"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = "2133.0"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Specialisté v oblasti ochrany životního prostředí " \
                   "(kromě průmyslové ekologie)"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = "5223.6"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Prodavači elektrotechniky, elektroniky a domácích potřeb"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

    def test_load_deu_de(self):
        crosswalk_name = "deu_de"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = "B 01104-104"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Offizier im Sanitätsdienst"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = "BE 3313"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Nicht akademische Fachkräfte im Rechnungswesen"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

    def test_load_dnk_da(self):
        crosswalk_name = "dnk_da"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = "http://data.europa.eu/esco/concept/danish-occupations/3359.3"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Fiskeribetjent"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = "http://data.europa.eu/esco/concept/danish-occupations/2149.26"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Koordinator"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

    def test_load_esp_es(self):
        crosswalk_name = "esp_es"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = "24531012"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "INGENIEROS PLANIFICADORES DE TRÁFICO"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = "38201026"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "DESARROLLADORES FRONT END (JUNIOR)"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

    def test_load_est_et(self):
        crosswalk_name = "est_et"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = "http://data.europa.eu/esco/concept/estonian-occupations/" \
               "3322.0001"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "müügiesindaja, müügikonsultant (va tehnika)"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = "http://data.europa.eu/esco/concept/estonian-occupations/" \
               "2434.0003"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "müügiesindaja (kommunikatsioonitehnoloogia)"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

    def test_load_fra_fr(self):
        crosswalk_name = "fra_fr"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = "15244"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Gemmologue"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = "17346"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Opérateur / Opératrice triage du réseau ferré"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

    def test_load_hrv_hr(self):
        crosswalk_name = "hrv_hr"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = "http://data.europa.eu/esco/concept/croatian-occupations/" \
               "2131.71.7"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Diplomirani inženjer/diplomirana inženjerka računalstva"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = "http://data.europa.eu/esco/concept/croatian-occupations/" \
               "7341.32.5"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "grafičar/grafičarka tiskarske proizvodnje, " \
                   "specijalizirani/specijalizirana"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

    def test_load_xx_xx(self):
        crosswalk_name = "xx_xx"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = ""
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = ""
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

    def test_load_hun_hu(self):
        crosswalk_name = "hun_hu"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = "http://data.europa.eu/esco/concept/hungarian-occupations/2534"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Informatikai és telekommunikációs technológiai termékek " \
                   "értékesítését tervező, szervező"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = "http://data.europa.eu/esco/isco/C2643"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Fordítók, tolmácsok és egyéb nyelvészek"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

    def test_load_ita_it(self):
        crosswalk_name = "ita_it"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = "1.1.1.1.0"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Membri di organismi di governo e di assemblee nazionali " \
                   "con potestà legislativa e regolamentare"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = "9.3.1.1.0"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Truppa delle forze armate"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

    def test_load_ltu_lt(self):
        crosswalk_name = "ltu_lt"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = "011001"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Generolas leitenantas {karinės sausumos pajėgos}"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = "962917"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Budėtojas"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

    def test_load_lva_lv(self):
        crosswalk_name = "lva_lv"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = "8342.01"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Bagarēšanas mašīnu OPERATORS"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = "2142"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Būvinženieri"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

    def test_load_nld_nl(self):
        crosswalk_name = "nld_nl"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = "1000409808"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Exportmanager"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = "2557"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Caddie"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

    def test_load_nor_no(self):
        crosswalk_name = "nor_no"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = "http://data.europa.eu/esco/concept/norwegian-occupations/9334"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Varepåfyllere"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = "http://data.europa.eu/esco/concept/norwegian-occupations/9213"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Hjelpearbeidere i kombinasjonsbruk"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

    def test_load_pol_pl(self):
        crosswalk_name = "pol_pl"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = "311410"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Technik mechatronik"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = "325511"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Technik ochrony środowiska"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

    def test_load_prt_pt(self):
        crosswalk_name = "prt_pt"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = "7212.1"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Soldador"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = "2619.1"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Conservador dos registos civil, automóvel, comercial " \
                   "e predial"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

    def test_load_rou_ro(self):
        crosswalk_name = "rou_ro"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = "http://data.europa.eu/esco/concept/romanian-occupation/211101"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "fizician"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = "http://data.europa.eu/esco/concept/romanian-occupation/311402"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "tehnician electronica"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

    def test_load_svk_sk(self):
        crosswalk_name = "svk_sk"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = "3240.003"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Inseminačný technik"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = "8111.004"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Lamač"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

    def test_load_svn_sl(self):
        crosswalk_name = "svn_sl"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = "5"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Častnik"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = "2378"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Delavec za druga preprosta dela"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

    def test_load_swe_sv(self):
        crosswalk_name = "swe_sv"
        crosswalk_loader = CrosswalkLoader(crosswalk_name=crosswalk_name)
        queries = crosswalk_loader.load()

        q_id = "http://data.jobtechdev.se/taxonomy/concept/h4wH_7kG_UTN"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Processoperatör, pappersmassa"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")

        q_id = "http://data.jobtechdev.se/taxonomy/concept/nDGY_Jbi_cRM"
        query_info = queries[q_id]

        actual = query_info["title"]
        expected = "Flygplatstekniker"
        self.assertEqual(actual, expected, "Wrong title.")

        actual = query_info["description"]
        expected = ""
        self.assertEqual(actual, expected, "Wrong description.")
