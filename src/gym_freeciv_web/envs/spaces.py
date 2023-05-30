from gymnasium.spaces import Dict, MultiBinary, MultiDiscrete
from gymnasium.spaces.utils import flatdim

def generate_action_space_info(max_cities=100, max_players=8, max_units=1000):
    action_space = Dict({
        #City interaction
        "select_city":  MultiBinary(max_cities, seed=42), #Allowing for maximum 100 cities to select from
        "city_prod_improv": MultiBinary(67, seed=30),
        "city_sell_improv": MultiBinary(67, seed=10),
        
        "city_prod_unit": MultiBinary(51, seed=10),
        "city_change_specialist": MultiBinary(20, seed=10),
        "city_buy_production": MultiBinary(1, seed=10),

        "city_unwork": MultiBinary(21),
        "city_work": MultiBinary(21),
        
        #Game interaction
        "game_interaction": MultiBinary(1),

        #Player interaction
        "self_interact": MultiBinary(4),
        
        "select_opponent": MultiBinary(max_players - 1, seed=32), #Allowing for maximum of 7 opponents
        "opponent_add_clause": MultiBinary(13),
        "opponent_remove_clause": MultiBinary(13),
        "opponent_cancel_clause": MultiBinary(6),
        "opponent_diplomacy": MultiBinary(3),
        "opponent_trade": MultiBinary(87),

        #Research interaction

        "research": MultiBinary(87),
        "tech_goal": MultiBinary(87),

        #Unit interaction

        "select_unit": MultiBinary(max_units, seed=32), #Allowing for maximum of 1000 units
        "unit_action": MultiBinary(30)
    })
    
    action_info = {
        #City interaction
        "select_city":  ["select_city_%i" for i in range(100)],
        "city_prod_improv": ["change_improve_prod_A.Smith's Trading Co._40", "change_improve_prod_Airport_0"],
        
        "city_prod_unit": ["change_unit_prod_AEGIS Cruiser_37", "change_unit_prod_AWACS_51"], #TODO - complete list
        "city_change_specialist": ["city_change_specialist_%i" % i for i in range(20)],
        "city_buy_production": ["buy_production?"], ###How to model this - buying production multiple times?

        "city_unwork": ["city_unwork_0_0_0", "city_unwork_10_0_-2", "city_unwork_11_0_2", "city_unwork_8_1_1", ""], #TODO - complete list
        
        #Game interaction
        "game_interaction": ["end_turn"],

        #Player interaction
        "self_interact": ["decrease_lux", "decrease_sci", "increase_lux", "increase_sci"], #Assuming player 0 = self
        
        "select_opponent": ["select_opponent%i" for i in range(1,8)], #Allowing for maximum of 7 opponents
        "opponent_add_clause": ["add_clause_clAlliance_player0", "add_clause_clAlliance_player1", "add_clause_clCeasefire_player0",
                            "add_clause_clCeasefire_player1"], #TODO - complete list
        "opponent_cancel_clause": ["cancel_clause_clAlliance_player0", "cancel_clause_clAlliance_player1", "cancel_clause_clCeasefire_player0", ""],
        "opponent_diplomacy": ["accept_treaty", "start_negotiation", "stop_negotiation"],
        "opponent_trade": ["trade_tech_clause_clAdvance_player0_10_Ceremonial Burial", ""],

        #Research interaction

        "research": ["research_tech_Advanced Flight_1", "" , "research_tech_Writing_87"],
        "tech_goal": ["set_tech_goal_Advanced Flight_1", "",  "set_tech_goal_Writing_87"],

        #Unit interaction

        "select_unit": ["select_unit%i" % i for i in range(1000)], #Allowing for maximum of 1000 units
        "unit_action": ["airbase", "airlift", "autosettlers", "build", "disband", "explore", "fallout", "forest", "fortify",
                "fortress", "goto_0", "goto_1", "goto_2", "goto_3", "goto_4", "goto_5", "goto_6", "goto_7", "homecity",
                "irrigation", "mine", "noorders", "paradrop", "pillage", "pollution", "railroad", "road", "transform",
                "unit_load", "unit_unload", "upgrade"]
    }
    action_info["city_sell_improv"] = [infostr.replace("change_improve_prod", "city_sell_improvement") for infostr in action_info["city_prod_improv"]]
    action_info["city_work"] = [infostr.replace("_unwork_", "_work_") for infostr in action_info["city_unwork"]]
    action_info["opponent_remove_clause"] = [infostr.replace("add", "remove") for infostr in action_info["opponent_add_clause"]]

    return action_space, action_info

def generate_observation_space(max_cities=100, max_units=1000, opponents=7):

    ospace = Dict({
        "city_improvements": MultiBinary([67, max_cities]),
        "city_shield_output": MultiDiscrete([10 for _ in range(max_cities * 21)]), #Max shield output is assumed 9 for all 21 city tiles times max_cities
        "city_food_output": MultiDiscrete([10 for _ in range(max_cities * 21)]), #Max food output is assumed 9 for all 21 city tiles times max_cities
        "city_trade_output": MultiDiscrete([10 for _ in range(max_cities * 21)]), #Max trade output is assumed 9 for all 21 city tiles times max_cities

        "city_bulbs": MultiDiscrete([100 for _ in range(max_cities)]),
        "city_corruption": MultiDiscrete([100 for _ in range(max_cities)]),
        "city_pollution": MultiDiscrete([100 for _ in range(max_cities)]),
        "city_waste": MultiDiscrete([100 for _ in range(max_cities)]),
        "food_stock": MultiDiscrete([100 for _ in range(max_cities)]),
        "granary_size": MultiDiscrete([100 for _ in range(max_cities)]),
        "granary_turn": MultiDiscrete([100 for _ in range(max_cities)]),
        "growth_in": MultiDiscrete([100 for _ in range(max_cities)]),
        "luxury": MultiDiscrete([100 for _ in range(max_cities)]),

        "ppl_angry": MultiDiscrete([100 for _ in range(max_cities)]),
        "ppl_content": MultiDiscrete([100 for _ in range(max_cities)]),
        "ppl_happy": MultiDiscrete([100 for _ in range(max_cities)]),
        "ppl_unhappy": MultiDiscrete([100 for _ in range(max_cities)]),
       
        "prod_food": MultiDiscrete([100 for _ in range(max_cities)]),
        "prod_gold": MultiDiscrete([100 for _ in range(max_cities)]),
        "prod_process": MultiDiscrete([100 for _ in range(max_cities)]),
        "prod_shield": MultiDiscrete([100 for _ in range(max_cities)]),
        "prod_trade": MultiDiscrete([100 for _ in range(max_cities)]),

        "production_kind": MultiDiscrete([100 for _ in range(max_cities)]),
        "production_value": MultiDiscrete([100 for _ in range(max_cities)]),

        "science": MultiDiscrete([100 for _ in range(max_cities)]),
        "size": MultiDiscrete([100 for _ in range(max_cities)]),
        "state": MultiDiscrete([100 for _ in range(max_cities)]),
        "surplus_food": MultiDiscrete([100 for _ in range(max_cities)]),
        "surplus_gold": MultiDiscrete([100 for _ in range(max_cities)]),
        "surplus_shield": MultiDiscrete([100 for _ in range(max_cities)]),
        "surplus_trade": MultiDiscrete([100 for _ in range(max_cities)]),
        "turns_to_prod_complete": MultiDiscrete([100 for _ in range(max_cities)]),

        "map_extras": MultiBinary([84, 56, 128]), #128 extra options for a 84x56 map
        "map_status": MultiBinary([84, 56,   5]), #5 status options for a 84x56 map
        "map_terrain": MultiBinary([84, 56, 16]), #16 terrain options for a 84x56 map
        
        "gov_type": MultiBinary(10), #TODO: Check hwo many government types are there -> Despotism -> Democracy
        "unit_can_transport": MultiBinary(max_units),
        "unit_health": MultiDiscrete([100 for _ in range(max_units)]),
        "unit_home_city": MultiDiscrete([max_cities+1 for _ in range(max_units)]),
        "unit_moves_left": MultiDiscrete([20 for _ in range(max_units)]),
        "unit_attack_strength": MultiDiscrete([100 for _ in range(max_units)]),
        
        "type_attack_strength": MultiDiscrete([100 for _ in range(max_units)]),
        "type_build_cost": MultiDiscrete([100 for _ in range(max_units)]),
        "type_convert_time": MultiDiscrete([100 for _ in range(max_units)]),
        "type_converted_to": MultiDiscrete([100 for _ in range(max_units)]),
        "type_defense_strength": MultiDiscrete([100 for _ in range(max_units)]),
        "type_firepower": MultiDiscrete([100 for _ in range(max_units)]),
        "type_hp": MultiDiscrete([100 for _ in range(max_units)]),
        "type_move_rate": MultiDiscrete([100 for _ in range(max_units)]),
        "type_rule_name": MultiDiscrete([100 for _ in range(max_units)]),
        "type_vision_radius_sq": MultiDiscrete([100 for _ in range(max_units)]),
        "type_worker": MultiBinary(max_units),
        "upkeep_food": MultiDiscrete([100 for _ in range(max_units)]),
        "upkeep_gold": MultiDiscrete([100 for _ in range(max_units)]),
        "upkeep_shield": MultiDiscrete([100 for _ in range(max_units)]),
        "veteran": MultiDiscrete([100 for _ in range(max_units)]),

        "opp_bulbs": MultiDiscrete([100 for _ in range(opponents)]),
        "opp_bulbs_researched": MultiDiscrete([1000 for _ in range(opponents)]),
        "opp_col_love": MultiBinary([10, opponents]), #TODO: check number of love types
        "opp_gold": MultiDiscrete([100 for _ in range(opponents)]),
        "opp_1gov": MultiBinary([10, opponents]), #TODO: check number of gov types
        "opp_inventions": MultiBinary([87, opponents]),
        "opp_luxury": MultiDiscrete([100 for _ in range(opponents)]),
        "opp_plr_score": MultiDiscrete([100 for _ in range(opponents)]),
        "opp_plr_type": MultiBinary([10, opponents]), #TODO: check type of player, e.g., Easy AI
        "opp_research": MultiBinary([87, opponents]),
        "opp_research_progress": MultiDiscrete([1000 for _ in range(opponents)]),
        "opp_researching_costs": MultiDiscrete([1000 for _ in range(opponents)]),
        "opp_science": MultiDiscrete([100 for _ in range(opponents)]),
        "opp_tax": MultiDiscrete([100 for _ in range(opponents)]),
        "opp_known": MultiBinary([213, opponents]), #TODO: Alternative to model unknown information - all 213 informations on opponent can be flagged unknown

        #TODO: Complete full observation overview - potentially construct them in the States themselves
    })
    
    return ospace

if __name__ == "__main__":
    aspace, ainfo = generate_action_space_info()
    ospace = generate_observation_space()

    print("Action space: Flat dimensionality: ", flatdim(aspace))
    print("Observaction space: Flat dimensionality: ", flatdim(ospace))
