import streamlit as st
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle

loaded_model= pickle.load(open('C:/Users/HP/Desktop/pet/trained_model.sav', 'rb'))

def pet_prediction(input_data):
    # Initialize LabelEncoders for categorical features
    pet_type_encoder = LabelEncoder()
    breed_encoder = LabelEncoder()
    color_encoder = LabelEncoder()
    size_encoder = LabelEncoder()

    # Fit the encoders with possible categories
    pet_type_encoder.fit(["Bird", "Rabbit", "Dog", "Cat"])
    breed_encoder.fit(["Parakeet", "Labrador", "Golden Retriever", "Poodle", "Persian", "Siamese"])
    color_encoder.fit(["Black", "Gray", "Brown", "White", "Orange"])
    size_encoder.fit(["Large", "Medium", "Small"])

    # Process input data
    processed_data = []
    for feature, value in input_data.items():
        if feature == "PetType":
            processed_data.append(pet_type_encoder.transform([value])[0])
        elif feature == "Breed":
            processed_data.append(breed_encoder.transform([value])[0])
        elif feature == "Color":
            processed_data.append(color_encoder.transform([value])[0])
        elif feature == "Size":
            processed_data.append(size_encoder.transform([value])[0])
        else:
            try:
                processed_data.append(float(value))
            except ValueError:
                processed_data.append(0)

    # Convert processed data to numpy array for prediction
    processed_data = np.array(processed_data).reshape(1, -1)

    return processed_data

def main():
    st.title('Pet Adoption Prediction')

    PetID = st.text_input('Unique identifier for each pet.')
    PetType = st.selectbox('PetType', ["Bird", "Rabbit", "Dog", "Cat"])
    Breed = st.selectbox('Specific breed of the pet.', ["Parakeet", "Labrador", "Golden Retriever", "Poodle", "Persian", "Siamese"])
    AgeMonths = st.slider('Age of the pet in months.', 0, 200)
    Color = st.selectbox('Color of the pet.', ["Black", "Gray", "Brown", "White", "Orange"])
    Size = st.selectbox('Size category of the pet (Small, Medium, Large).', ["Large", "Medium", "Small"])
    WeightKg = st.text_input('Weight of the pet in kilograms')
    Vaccinated = st.selectbox('Vaccination status of the pet (0 - Not vaccinated, 1 - Vaccinated)', ['0', '1'])
    HealthCondition = st.selectbox('Health condition of the pet (0 - Healthy, 1 - Medical condition)', ['0', '1'])
    TimeInShelterDays = st.text_input(' Duration the pet has been in the shelter (days)')
    AdoptionFee = st.text_input(' Adoption fee charged for the pet (in dollars).')
    PreviousOwner = st.selectbox('Whether the pet had a previous owner (0 - No, 1 - Yes)', ['0', '1'])

    diagnosis = ''

    if st.button('Pet adoption result'):
        input_data = {
            "PetID": PetID,
            "PetType": PetType,
            "Breed": Breed,
            "AgeMonths": AgeMonths,
            "Color": Color,
            "Size": Size,
            "WeightKg": WeightKg,
            "Vaccinated": Vaccinated,
            "HealthCondition": HealthCondition,
            "TimeInShelterDays": TimeInShelterDays,
            "AdoptionFee": AdoptionFee,
            "PreviousOwner": PreviousOwner
        }
        processed_data = pet_prediction(input_data)
        
        
        #predictions = np.random.choice([0, 1], size=(1,))  
        predictions= loaded_model.predict(processed_data)

        if predictions[0] == 0:
            diagnosis = 'Pet is unlikely to be adopted'
        else:
            diagnosis = 'Pet is likely to be adopted'

        st.success(diagnosis)

if __name__ == '__main__':
    main()
