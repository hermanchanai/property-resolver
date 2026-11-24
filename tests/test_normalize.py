from tools.normalize import address_relation, name_key, name_match, name_sim, parse_address, phase_marker


def test_name_key_drops_marketing_words_and_romanizes():
    assert name_key("The Palms Apartments II") == "palms 2"


def test_phase_marker():
    assert phase_marker("Encantada Phase II") == "2"
    assert phase_marker("Villas at Elm III") == "3"
    assert phase_marker("Liv Crossroads") is None


def test_name_sim_ignores_marketing_words():
    assert name_sim("Liv Crossroads Apartments", "LIV CROSSROADS") == 1.0
    assert name_sim("Apartments", "Liv") == 0.0


def test_name_match_rule():
    assert name_match("Nines at Lakeside", "The Nines Lakeside Apartments")
    assert not name_match("Verve", "Vista")


def test_parse_address_key_and_zip():
    a = parse_address("10401 N 52nd Street, Paradise Valley, AZ 85253-1234")
    assert (a.number, a.street, a.zip5) == ("10401", "52", "85253")
    b = parse_address("10401 N 52ND ST")
    assert a.key == b.key


def test_address_relation():
    a = parse_address("4240 E Campbell Ave, Phoenix, AZ 85018")
    assert address_relation(a, parse_address("4240 E CAMPBELL AVE PHOENIX 85018")) == "exact"
    assert address_relation(a, parse_address("4300 E Campbell Ave")) == "street"
    assert address_relation(a, parse_address("4240 E Thomas Rd")) == "none"
    assert address_relation(a, parse_address("")) == "na"
