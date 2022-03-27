import random
import numpy as np
import torch
import re
from seqeval.metrics import f1_score as seqeval_f1_score
from seqeval.scheme import IOB2
from sklearn.metrics import f1_score as sklearn_f1_score
from transformers import EvalPrediction
from datasets import load_metric


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slot_metrics(labels, preds):
    f1 = seqeval_f1_score(y_true=labels, y_pred=preds, mode='strict', scheme=IOB2)
    # Use the following instead for the ATIS dataset
    # f1 = seqeval_f1_score(y_true=labels, y_pred=preds)

    return f1


def intent_metrics(labels, preds):
    labels, preds = np.array(labels), np.array(preds)
    accuracy = (labels == preds).mean()
    f1 = sklearn_f1_score(y_true=labels, y_pred=preds, average='weighted')

    return accuracy, f1


def convert_t5_output_to_slot_preds(pred):
    """ Converts raw prediction into slot prediction using the T5 sentinel tokens (<extra_id_0>, <extra_id_1> etc.)"""
    match = re.split(r"<extra_id_\d+>", pred)
    if match:
        # Skip the <pad> and </s> (eos) tokens at the start and the end of the output respectively
        return match[1:-1]


def compute_micro_f1(scores: dict):
    tps = 0  # true positives
    fps = 0  # false positives
    fns = 0  # false negatives
    for slot_scores in scores.values():
        tps += slot_scores["true_positives"]
        fps += slot_scores["false_positives"]
        fns += slot_scores["false_negatives"]

    print(f"# of True Positives= {tps}\t# of False Positives= {fps}\t# of False Negatives= {fns}")
    epsilon = 1e-10  # used to prevent division by zero
    micro_precision = tps / (tps + fps + epsilon)
    micro_recall = tps / (tps + fns + epsilon)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + epsilon)

    return micro_precision, micro_recall, micro_f1


# To be used for mask filling
INTENT_MAPPING = {
    "UNK": "unknown",
    "AddToPlaylist": "playlist",
    "BookRestaurant": "restaurant",
    "GetWeather": "weather",
    "PlayMusic": "music",
    "RateBook": "book",
    "SearchCreativeWork": "creative work",
    "SearchScreeningEvent": "movie"
}

SLOT_MAPPING = {
    "UNK": "unknown",
    "album": "album",
    "artist": "artist",
    "best_rating": "best rating",
    "city": "city",
    "condition_description": "weather condition",
    "condition_temperature": "temperature condition",
    "country": "country",
    "cuisine": "cuisine",
    "current_location": "current location",
    "entity_name": "track name",
    "facility": "facility",
    "genre": "genre",
    "geographic_poi": "geographic area",
    "location_name": "location name",
    "movie_name": "movie name",
    "movie_type": "movie type",
    "music_item": "music item",
    "object_location_type": "location type",
    "object_name": "object name",
    "object_part_of_series_type": "type of series",
    "object_select": "selected object",
    "object_type": "object type",
    "party_size_description": "people description",
    "party_size_number": "number of people",
    "playlist": "playlist",
    "playlist_owner": "playlist owner",
    "poi": "point of interest",
    "rating_unit": "rating unit",
    "rating_value": "rating value",
    "restaurant_name": "name of the restaurant",
    "restaurant_type": "type of restaurant",
    "served_dish": "dish",
    "service": "service",
    "sort": "sort",
    "spatial_relation": "relative place",  # approximate area
    "state": "state",
    "timeRange": "time range",
    "track": "track to play",
    "year": "year"
}

# SLOT_MAPPING = {
#     "UNK": "unknown",
#     "album": "name of the album",
#     "artist": "name of the artist",
#     "best_rating": "best possible rating",
#     "city": "name of the city",
#     "condition_description": "weather",
#     "condition_temperature": "temperature",
#     "country": "name of the country",
#     "cuisine": "type of the cuisine",
#     "current_location": "current location",
#     "entity_name": "name of the track",
#     "facility": "facility",
#     "genre": "genre of the track",
#     "geographic_poi": "geographic point",
#     "location_name": "name of the location",
#     "movie_name": "name of the movie",
#     "movie_type": "type of the movie",
#     "music_item": "music item",
#     "object_location_type": "type of the location",
#     "object_name": "name of the object of concern",
#     "object_part_of_series_type": "object is part of series type",
#     "object_select": "object to select",
#     "object_type": "type of the object",
#     "party_size_description": "description of the people",
#     "party_size_number": "size of people",
#     "playlist": "name of the playlist",
#     "playlist_owner": "name of the playlist owner",
#     "poi": "point of interest",
#     "rating_unit": "unit of the rating",
#     "rating_value": "value of the rating",
#     "restaurant_name": "name of the restaurant",
#     "restaurant_type": "type of the restaurant",
#     "served_dish": "dish that is served",
#     "service": "type of service",
#     "sort": "sort",
#     "spatial_relation": "approximate area",
#     "state": "name of the state",
#     "timeRange": "time range",
#     "track": "track to play",
#     "year": "year"
# }

INVERTED_SLOT_MAPPING = {
    v: k for k, v in SLOT_MAPPING.items()
}


ATIS_INTENT_MAPPING = {
    'atis_flight': 'flight',
    'atis_airfare': 'fare',
    'atis_ground_service': 'ground service',
    'atis_airline': 'airline',
    'atis_abbreviation': 'abbreviation',
    'atis_aircraft': 'aircraft',
    'atis_flight_time': 'flight time',
    'atis_quantity': 'quantity',
    'atis_flight#atis_airfare': 'flight and fare',
    'atis_city': 'city',
    'atis_distance': 'distance',
    'atis_airport': 'airport',
    'atis_ground_fare': 'ground fare',
    'atis_capacity': 'capacity',
    'atis_flight_no': 'flight number',
    'atis_meal': 'meal',
    'atis_restriction': 'restriction',
    'atis_airline#atis_flight_no': 'airline and flight number',
    'atis_aircraft#atis_flight#atis_flight_no': 'aircraft and flight and flight number',
    'atis_cheapest': 'cheapest',
    'atis_ground_service#atis_ground_fare': 'ground service and fare'
}


# ATIS_SLOT_MAPPING = {
ATIS_SLOTS = {
    'aircraft_code': '',
    'airline_code': '',
    'airline_name': '',
    'airport_code': '',
    'airport_name': '',
    'arrive_date.date_relative': '',
    'arrive_date.day_name': '',
    'arrive_date.day_number': '',
    'arrive_date.month_name': '',
    'arrive_date.today_relative': '',
    'arrive_time.end_time': '',
    'arrive_time.period_mod': '',
    'arrive_time.period_of_day': '',
    'arrive_time.start_time': '',
    'arrive_time.time': '',
    'arrive_time.time_relative': '',
    'city_name': '',
    'class_type': '',
    'connect': '',
    'cost_relative': '',
    'day_name': '',
    'day_number': '',
    'days_code': '',
    'depart_date.date_relative': '',
    'depart_date.day_name': '',
    'depart_date.day_number': '',
    'depart_date.month_name': '',
    'depart_date.today_relative': '',
    'depart_date.year': '',
    'depart_time.end_time': '',
    'depart_time.period_mod': '',
    'depart_time.period_of_day': '',
    'depart_time.start_time': '',
    'depart_time.time': '',
    'depart_time.time_relative': '',
    'economy': '',
    'fare_amount': '',
    'fare_basis_code': '',
    'flight_days': '',
    'flight_mod': '',
    'flight_number': '',
    'flight_stop': '',
    'flight_time': '',
    'fromloc.airport_code': '',
    'fromloc.airport_name': '',
    'fromloc.city_name': '',
    'fromloc.state_code': '',
    'fromloc.state_name': '',
    'meal': '',
    'meal_code': '',
    'meal_description': '',
    'mod': '',
    'month_name': '',
    'or': '',
    'period_of_day': '',
    'restriction_code': '',
    'return_date.date_relative': '',
    'return_date.day_name': '',
    'return_date.day_number': '',
    'return_date.month_name': '',
    'return_date.today_relative': '',
    'return_time.period_mod': '',
    'return_time.period_of_day': '',
    'round_trip': '',
    'state_code': '',
    'state_name': '',
    'stoploc.airport_name': '',
    'stoploc.city_name': '',
    'stoploc.state_code': '',
    'time': '',
    'time_relative': '',
    'today_relative': '',
    'toloc.airport_code': '',
    'toloc.airport_name': '',
    'toloc.city_name': '',
    'toloc.country_name': '',
    'toloc.state_code': '',
    'toloc.state_name': '',
    'transport_type': ''
}

ATIS_SLOT_MAPPING = {
    key: ' '.join(re.split('_|\.', key))
    for key in ATIS_SLOTS.keys()
}
