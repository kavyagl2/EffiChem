import numpy as np
import pandas as pd
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, Crippen, Lipinski, QED

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# RDKit Descriptors Extraction
def calculate_rdkit_features(smiles):
    """Calculate selected RDKit molecular descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
     
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol)

    return {
        "Molecular Weight": Descriptors.MolWt(mol),  # Precise molecular mass
        "ALogP": Crippen.MolLogP(mol),  # Octanol-water partition coefficient
        "Molar Refractivity": Chem.Crippen.MolMR(mol),  # Measure of total polarizability
        "Heavy Atom Count": Lipinski.HeavyAtomCount(mol),  # Non-hydrogen atoms
        "Atom Count": mol.GetNumAtoms(),  # Total number of atoms
        "Bond Count": mol.GetNumBonds(),  # Total number of chemical bonds
        "Ring Count": rdMolDescriptors.CalcNumRings(mol),  # Total number of rings
        "Aromatic Ring Count": rdMolDescriptors.CalcNumAromaticRings(mol),  # Number of aromatic rings
        "Saturated Ring Count": rdMolDescriptors.CalcNumSaturatedRings(mol),  # Number of non-aromatic rings
        "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),  # Atoms that can accept hydrogen bonds
        "H-Bond Donors": rdMolDescriptors.CalcNumHBD(mol),  # Atoms that can donate hydrogen bonds
        "Topological Polar Surface Area": rdMolDescriptors.CalcTPSA(mol),  # Surface area of polar atoms
        "Rotatable Bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),  # Bonds that can freely rotate
        "Formal Charge": AllChem.GetFormalCharge(mol),  # Overall electric charge of the molecule
        #"Molecular Volume": AllChem.ComputeMolVolume(mol),  # Approximate 3D molecular volume
        "QED (Drug-likeness)": QED.qed(mol),  # Quantitative estimate of drug-likeness     
        "Radical Electrons": Descriptors.NumRadicalElectrons(mol),  # Unpaired electrons
        "sp3 Carbon Fraction": rdMolDescriptors.CalcFractionCSP3(mol),  # Fraction of sp3 hybridized carbons
    }

def extract_features(df):
    """Apply RDKit feature extraction to the dataset."""
    logging.info("Extracting RDKit features...")

    df["RDKit_Features"] = df["Canonicalized SMILES"].apply(calculate_rdkit_features)

    # Convert dictionary column to individual feature columns
    features_df = df["RDKit_Features"].apply(pd.Series)
    df = pd.concat([df.drop(columns=["RDKit_Features"]), features_df], axis=1)

    missing_rdkit = df.isnull().sum().sum()
    logging.info(f"Missing RDKit features: {missing_rdkit}")

    logging.info(f"Feature extraction complete. Total samples: {len(df)}")
    return df 
