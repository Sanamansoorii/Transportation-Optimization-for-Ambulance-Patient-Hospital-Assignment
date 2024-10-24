!pip install pandas numpy scikit-learn pulp seaborn matplotlib
!pip install pyomo
!apt-get install -y -qq glpk-utils  # Install GLPK solver for Pyomo

# Import necessary libraries
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from google.colab import files

# Step 1: Generate synthetic data for patients, ambulances, and hospitals
np.random.seed(42)
num_patients = 20
num_ambulances = 20  # Equal to the number of patients
num_hospitals = 6

# Synthetic data for patients
patients_data = {
    'Patient_ID': np.arange(1, num_patients + 1),
    'Patient_Location_X': np.random.uniform(0, 100, num_patients),
    'Patient_Location_Y': np.random.uniform(0, 100, num_patients),
    'Patient_Condition': np.random.randint(1, 11, num_patients)  # Patient condition/severity
}
patients_df = pd.DataFrame(patients_data)

# Synthetic data for ambulances
ambulances_data = {
    'Ambulance_ID': np.arange(1, num_ambulances + 1),
    'Ambulance_Location_X': np.random.uniform(0, 100, num_ambulances),
    'Ambulance_Location_Y': np.random.uniform(0, 100, num_ambulances),
    'Ambulance_Availability': np.ones(num_ambulances)  # All ambulances are available
}
ambulances_df = pd.DataFrame(ambulances_data)

# Synthetic data for hospitals
hospitals_data = {
    'Hospital_ID': np.arange(1, num_hospitals + 1),
    'Hospital_Location_X': np.random.uniform(0, 100, num_hospitals),
    'Hospital_Location_Y': np.random.uniform(0, 100, num_hospitals),
    'Hospital_Capacity': np.random.randint(4, 8, num_hospitals)  # Adjusted capacities
}
hospitals_df = pd.DataFrame(hospitals_data)

# Cost per distance unit
cost_per_distance = 10

# Step 2: Assign each ambulance to a patient based on distance
def calculate_cost_matrix_ambulances_patients(ambulances_df, patients_df):
    distances = np.sqrt((ambulances_df['Ambulance_Location_X'].values[:, None] - patients_df['Patient_Location_X'].values)**2 +
                        (ambulances_df['Ambulance_Location_Y'].values[:, None] - patients_df['Patient_Location_Y'].values)**2)
    costs = distances * cost_per_distance
    return costs

ambulance_patient_cost = calculate_cost_matrix_ambulances_patients(ambulances_df, patients_df)

# Objective function: Minimize total distance between ambulances and patients
c = ambulance_patient_cost.flatten()

# Constraints
# Each ambulance must be assigned to exactly one patient
A_eq = np.zeros((num_ambulances, num_ambulances * num_patients))
for i in range(num_ambulances):
    A_eq[i, i*num_patients:(i+1)*num_patients] = 1
b_eq = np.ones(num_ambulances)

# Each patient must be assigned to exactly one ambulance
A_ub = np.zeros((num_patients, num_ambulances * num_patients))
for j in range(num_patients):
    A_ub[j, j::num_patients] = 1
b_ub = np.ones(num_patients)

# Bounds
x_bounds = [(0, 1)] * (num_ambulances * num_patients)

# Solve the linear programming problem for ambulance-patient assignment
result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=x_bounds, method='highs')

# Extract the results
assignment_matrix = result.x.reshape((num_ambulances, num_patients))

# Create a DataFrame for ambulance-patient assignments
ambulance_patient_assignments = []
for i in range(num_ambulances):
    patient_id = np.argmax(assignment_matrix[i, :]) + 1
    cost = ambulance_patient_cost[i, patient_id - 1]
    ambulance_patient_assignments.append({
        'Ambulance_ID': ambulances_df.loc[i, 'Ambulance_ID'],
        'Patient_ID': patient_id,
        'Ambulance_Location_X': ambulances_df.loc[i, 'Ambulance_Location_X'],
        'Ambulance_Location_Y': ambulances_df.loc[i, 'Ambulance_Location_Y'],
        'Patient_Location_X': patients_df.loc[patient_id - 1, 'Patient_Location_X'],
        'Patient_Location_Y': patients_df.loc[patient_id - 1, 'Patient_Location_Y'],
        'Ambulance_to_Patient_Cost': cost
    })

ambulance_patient_df = pd.DataFrame(ambulance_patient_assignments)

# Step 3: Assign each patient to a hospital considering capacity and distance
def calculate_cost_matrix_patients_hospitals(patients_df, hospitals_df):
    distances = np.sqrt((patients_df['Patient_Location_X'].values[:, None] - hospitals_df['Hospital_Location_X'].values)**2 +
                        (patients_df['Patient_Location_Y'].values[:, None] - hospitals_df['Hospital_Location_Y'].values)**2)
    costs = distances * cost_per_distance
    return costs

patient_hospital_cost = calculate_cost_matrix_patients_hospitals(patients_df, hospitals_df)

# Objective function: Minimize total distance between patients and hospitals
c = patient_hospital_cost.flatten()

# Constraints
# Each patient must be assigned to exactly one hospital
A_eq = np.zeros((num_patients, num_patients * num_hospitals))
for i in range(num_patients):
    A_eq[i, i*num_hospitals:(i+1)*num_hospitals] = 1
b_eq = np.ones(num_patients)

# Each hospital has a capacity
A_ub = np.zeros((num_hospitals, num_patients * num_hospitals))
for j in range(num_hospitals):
    A_ub[j, j::num_hospitals] = 1
b_ub = hospitals_df['Hospital_Capacity'].values

# Bounds
x_bounds = [(0, 1)] * (num_patients * num_hospitals)

# Solve the linear programming problem for patient-hospital assignment
result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=x_bounds, method='highs')

# Extract the results
assignment_matrix = result.x.reshape((num_patients, num_hospitals))

# Create a DataFrame for patient-hospital assignments
patient_hospital_assignments = []
for i in range(num_patients):
    hospital_id = np.argmax(assignment_matrix[i, :]) + 1
    cost = patient_hospital_cost[i, hospital_id - 1]
    patient_hospital_assignments.append({
        'Patient_ID': patients_df.loc[i, 'Patient_ID'],
        'Hospital_ID': hospital_id,
        'Patient_Location_X': patients_df.loc[i, 'Patient_Location_X'],
        'Patient_Location_Y': patients_df.loc[i, 'Patient_Location_Y'],
        'Hospital_Location_X': hospitals_df.loc[hospital_id - 1, 'Hospital_Location_X'],
        'Hospital_Location_Y': hospitals_df.loc[hospital_id - 1, 'Hospital_Location_Y'],
        'Patient_to_Hospital_Cost': cost
    })

patient_hospital_df = pd.DataFrame(patient_hospital_assignments)

# Step 4: Combine the assignments with cost details
# Merge ambulance-patient assignments with patient-hospital assignments
full_assignments_df = pd.merge(ambulance_patient_df, patient_hospital_df, on='Patient_ID', suffixes=('_Ambulance_Patient', '_Patient_Hospital'))

# Calculate total cost
full_assignments_df['Total_Cost'] = full_assignments_df['Ambulance_to_Patient_Cost'] + full_assignments_df['Patient_to_Hospital_Cost']

# Add more details to the assignments DataFrame
full_assignments_df['Patient_Condition'] = patients_df['Patient_Condition']
full_assignments_df['Hospital_Capacity'] = full_assignments_df['Hospital_ID'].apply(lambda x: hospitals_df.loc[hospitals_df['Hospital_ID'] == x, 'Hospital_Capacity'].values[0])

# Save the detailed assignments to CSV
full_assignments_df.to_csv('detailed_ambulance_patient_hospital_assignments.csv', index=False)
print("Detailed Ambulance-Patient-Hospital assignments saved to detailed_ambulance_patient_hospital_assignments.csv")

# Step 5: Visualize the assignments
plt.figure(figsize=(14, 8))

# Plot patients
plt.scatter(patients_df['Patient_Location_X'], patients_df['Patient_Location_Y'], c='blue', label='Patients', s=50, edgecolor='k')
for i, row in patients_df.iterrows():
    plt.text(row['Patient_Location_X'], row['Patient_Location_Y'], f'P{row["Patient_ID"]}', fontsize=9, ha='right', color='blue')

# Plot ambulances
plt.scatter(ambulances_df['Ambulance_Location_X'], ambulances_df['Ambulance_Location_Y'], c='red', label='Ambulances', s=50, edgecolor='k')
for i, row in ambulances_df.iterrows():
    plt.text(row['Ambulance_Location_X'], row['Ambulance_Location_Y'], f'A{row["Ambulance_ID"]}', fontsize=9, ha='right', color='red')

# Plot hospitals
plt.scatter(hospitals_df['Hospital_Location_X'], hospitals_df['Hospital_Location_Y'], c='green', label='Hospitals', s=80, edgecolor='k')
for i, row in hospitals_df.iterrows():
    plt.text(row['Hospital_Location_X'], row['Hospital_Location_Y'], f'H{row["Hospital_ID"]}', fontsize=9, ha='right', color='green')

# Plot connections
for i, row in full_assignments_df.iterrows():
    plt.plot([row['Ambulance_Location_X'], row['Patient_Location_X']], [row['Ambulance_Location_Y'], row['Patient_Location_Y']], 'gray', linestyle='--', alpha=0.5)
    plt.plot([row['Patient_Location_X'], row['Hospital_Location_X']], [row['Patient_Location_Y'], row['Hospital_Location_Y']], 'gray', linestyle='--', alpha=0.5)

plt.title('Ambulance-Patient-Hospital Assignments')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid()
plt.savefig('ambulance_patient_hospital_assignments.png')
plt.show()

# Download the files
files.download('detailed_ambulance_patient_hospital_assignments.csv')
files.download('ambulance_patient_hospital_assignments.png')