import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

age = ctrl.Antecedent(np.arange(10, 101, 1), 'age')
tumor_size = ctrl.Antecedent(np.arange(0, 61, 1), 'tumor_size')
inv_nodes = ctrl.Antecedent(np.arange(0, 41, 1), 'inv_nodes')
deg_malig = ctrl.Antecedent(np.arange(1, 4, 1), 'deg_malig')
recurrence = ctrl.Consequent(np.arange(0, 101, 1), 'recurrence')


#Age membership functions
age['young'] = fuzz.trapmf(age.universe, [10, 10, 25, 50])
age['middle_aged'] = fuzz.trimf(age.universe, [25, 50, 70])
age['elderly'] = fuzz.trapmf(age.universe, [50, 70, 100, 100])

#Tumor size membership function
tumor_size['small'] = fuzz.trapmf(tumor_size.universe, [0, 0, 2, 10])
tumor_size['medium'] = fuzz.trimf(tumor_size.universe, [2, 10, 18])
tumor_size['large'] = fuzz.trapmf(tumor_size.universe, [10, 18, 60, 60])

#inv nodes membership function
inv_nodes['few'] = fuzz.trapmf(inv_nodes.universe, [0, 0, 2, 10])
inv_nodes['moderate'] = fuzz.trimf(inv_nodes.universe, [2, 10, 20])
inv_nodes['many'] = fuzz.trapmf(inv_nodes.universe, [10, 20, 40, 40])

#deg_malig membership function
deg_malig['low'] = fuzz.trimf(deg_malig.universe, [1, 1, 4])
deg_malig['high'] = fuzz.trimf(deg_malig.universe, [2, 4, 4])

# Define membership functions for the output (recurrence)
recurrence['low'] = fuzz.trimf(recurrence.universe, [0, 0, 50])
recurrence['medium'] = fuzz.trimf(recurrence.universe, [0, 50, 100])
recurrence['high'] = fuzz.trimf(recurrence.universe, [50, 100, 100])


# Define fuzzy rules based on the membership functions and reasoning
rule =[]
rule.append(ctrl.Rule(age['young'], recurrence['high']))
rule.append(ctrl.Rule(age['middle_aged'], recurrence['medium']))
rule.append(ctrl.Rule(age['elderly'], recurrence['medium']))

rule.append(ctrl.Rule(inv_nodes['few'], recurrence['low']))
rule.append(ctrl.Rule(inv_nodes['moderate'], recurrence['medium']))
rule.append(ctrl.Rule(inv_nodes['many'], recurrence['high']))

rule.append(ctrl.Rule(deg_malig['low'], recurrence['low']))
rule.append(ctrl.Rule(deg_malig['high'], recurrence['high']))

rule.append(ctrl.Rule(tumor_size['small'], recurrence['low']))
rule.append(ctrl.Rule(tumor_size['medium'], recurrence['medium']))
rule.append(ctrl.Rule(tumor_size['large'], recurrence['high']))

rule.append(ctrl.Rule(inv_nodes['few'] & tumor_size['large'], recurrence['medium']))
rule.append(ctrl.Rule(deg_malig['high'] & tumor_size['large'], recurrence['high']))
rule.append(ctrl.Rule(age['middle_aged'] & inv_nodes['few'], recurrence['low']))
rule.append(ctrl.Rule(age['middle_aged'] & inv_nodes['moderate'], recurrence['low']))

#add the rules to the control system
recurrence_ctrl = ctrl.ControlSystem(rules=rule)

# Create simulation
recurrence_sim = ctrl.ControlSystemSimulation(recurrence_ctrl)

# Initialize inputs
recurrence_sim.input['age'] = 10
recurrence_sim.input['tumor_size'] = 1
recurrence_sim.input['inv_nodes'] = 1
recurrence_sim.input['deg_malig'] = 1

# Get input
def get_valid_input(prompt, min_value, max_value):
    while True:
        try:
            user_input = float(input(prompt))
            if min_value <= user_input <= max_value:
                return user_input
            else:
                print(f"Please enter a number between {min_value} and {max_value}.")
        except ValueError:
            print("Please enter a valid number.")

recurrence_sim.input['age'] = get_valid_input("Enter age (between 10 and 100): ", 10, 100)
recurrence_sim.input['tumor_size'] = get_valid_input("Enter tumor size (between 0 and 60): ", 0, 60)
recurrence_sim.input['inv_nodes'] = int(get_valid_input("Enter number of involved nodes (between 0 and 40): ", 0, 40))
recurrence_sim.input['deg_malig'] = int(get_valid_input("Enter degree of malignancy (between 1 and 3): ", 1, 3))

# Compute output
recurrence_sim.compute()
fuzzy_output_demo = recurrence_sim.output['recurrence']
crisp_output_demo = "recurrence" if fuzzy_output_demo>50 else "no recurrence" 
# Access output
print(f"The patient will experience {crisp_output_demo} at with score of {fuzzy_output_demo}.")