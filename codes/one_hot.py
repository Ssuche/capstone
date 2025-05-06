import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def splits():
    # load the data
    data = pd.read_csv("../processed_data/merged_data.csv")
    list_bp = ['avg_dbp', 'avg_diff', 'avg_sbp', 'max_sbp']
    list_ed = ['age', 'sex', 'language', 'insurance_type', 'primary_care', 
                'ed_name', 'bpa_response', 'htn_on_pl', 'htn_on_pmh', 
                'hld_on_pl', 'hld_on_pmh', 'family_dm', 'tobacco_user', 
                'htn_meds', 'statin_meds', 'disposition', 'detailed_race', 
                'weight', 'bmi', 'hba1c', 'height', 'sbp_1st', 'dbp_1st', 
                'poct_gluc']
    list_lab = [ 'avg_value_CHOLESTEROL, TOTAL', 'avg_value_CREATININE', 'avg_value_GLUCOSE', 'avg_value_GLUCOSE, POC', 'avg_value_HDL CHOLESTEROL', 'avg_value_HEMOGLOBIN A1C', 'avg_value_LDL CHOLESTEROL, CALCULATED']
    list_geo = [
        'po_box', 'homeless', 'total_pop', 'households', 'housing_units', 
        'p_children', 'p_elderly', 'p_adults', 'p_female', 'mdn_age', 
        'p_nhwhite', 'p_nhblack', 'p_hispanic', 'p_nhasian', 'p_other', 
        'p_moved', 'p_longcommute', 'p_marriednone', 'p_marriedkids', 
        'p_singlenone', 'p_malekids', 'p_femalekids', 'p_cohabitkids', 
        'p_nohsdeg', 'p_hsonly', 'p_somecollege', 'p_collegeplus', 
        'p_onlyenglish', 'p_spanishlimited', 'p_asianlimited', 'p_otherlimited', 
        'p_limitedall', 'p_notlimited', 'p_popbelow1fpl', 'p_popbelow2fpl', 
        'p_povmarriedfam', 'p_povmalefam', 'p_povfemalefam', 'hh_mdnincome', 
        'p_pubassist', 'p_foodstamps', 'p_assistorfood', 'p_unemployed', 
        'h_vacant', 'h_renter', 'h_occupants', 'h_novehicles', 'h_mdnrent', 
        'h_rentpercent', 'h_houseprice', 'p_private', 'p_medicare', 'p_medicaid', 
        'p_otherinsur', 'p_uninsured', 'h_nointernet', 'h_nocomputer', 
        'p_foreign', 'p_disabled']
    list_visit = ['visit_type']
    lists = list_bp + list_ed + list_lab + list_geo + list_visit
    X_all = data[lists]
    y = data['pcp_followup'].map({'Yes': 1, 'No': 0})
    y = np.array(y).astype(int)

    # encode
    numeric_cols = X_all.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X_all.select_dtypes(exclude=['number']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    X_preprocessed = preprocessor.fit_transform(X_all)
    if hasattr(X_preprocessed, "toarray"):
        X_preprocessed = X_preprocessed.toarray()
    num_feature_names = np.array(numeric_cols)  
    onehot = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = onehot.get_feature_names_out(categorical_cols)
    all_feature_names = np.concatenate([num_feature_names, cat_feature_names])
    
    # select features
    selected_feature_names = [
        'poct_gluc',
        'p_povfemalefam',
        'h_nointernet',
        'p_hsonly',
        'p_nohsdeg',
        'p_singlenone',
        'p_children',
        'p_unemployed',
        'p_foreign',
        'p_foodstamps',
        'p_assistorfood',
        'p_uninsured',
        'p_female',
        'total_pop',
        'p_popbelow1fpl',
        'p_popbelow2fpl',
        'insurance_type_MEDICAID',
        'primary_care_YES',
        'detailed_race_Black',
        'insurance_type_SELFPAY',
        
        
        'ed_name_NYU COBBLE HILL',
        'hld_on_pl_Yes',
        'statin_meds_Yes',
        'sex_Female',
        'language_English',
        
    ]
    
    X_preprocessed = pd.DataFrame(X_preprocessed, columns=all_feature_names)
    X_selected = X_preprocessed[selected_feature_names]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=50, stratify=y)
    
    return X_train, X_test, y_train, y_test