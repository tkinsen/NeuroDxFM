from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class AccessLevel(StrEnum):
    APPLICATION = "application"
    REGISTRATION = "registration"
    OPEN = "open"


class CohortRole(StrEnum):
    PRETRAINING = "pretraining"
    DEVELOPMENT = "development"
    CROSS_SITE = "cross_site"
    PRESYMPTOMATIC = "presymptomatic"
    CROSS_DISEASE = "cross_disease"
    ROBUSTNESS = "robustness"
    REFERENCE = "reference"


@dataclass(frozen=True)
class CohortDefinition:
    key: str
    display_name: str
    release: str
    role: CohortRole
    participants: int
    modalities: tuple[str, ...]
    diagnoses: tuple[str, ...]
    vendors: tuple[str, ...]
    field_strengths: tuple[str, ...]
    access: AccessLevel
    manifest_name: str
    expected_columns: tuple[str, ...]


BASE_COLUMNS = (
    "subject_id",
    "visit_id",
    "cohort",
    "t1",
    "diagnosis",
    "protocol",
    "months",
    "apoe4",
    "site",
    "vendor",
    "field_strength",
    "sequence",
)


BIOMARKER_COLUMNS = (
    "flair",
    "fdg_pet",
    "amyloid_pet",
    "csf",
    "amyloid",
    "anatomy",
)


UK_BIOBANK = CohortDefinition(
    key="ukb",
    display_name="UK Biobank",
    release="brain-imaging-current",
    role=CohortRole.PRETRAINING,
    participants=50000,
    modalities=("t1", "flair", "dwi", "rsfmri"),
    diagnoses=("population",),
    vendors=("siemens",),
    field_strengths=("3T",),
    access=AccessLevel.APPLICATION,
    manifest_name="ukb.csv",
    expected_columns=BASE_COLUMNS + BIOMARKER_COLUMNS,
)


ADNI = CohortDefinition(
    key="adni",
    display_name="ADNI-1/GO/2/3/4",
    release="ADNI4",
    role=CohortRole.DEVELOPMENT,
    participants=2400,
    modalities=("t1", "flair", "fdg_pet", "amyloid_pet", "tau_pet", "csf"),
    diagnoses=("cn", "mci", "ad"),
    vendors=("siemens", "ge", "philips"),
    field_strengths=("1.5T", "3T"),
    access=AccessLevel.APPLICATION,
    manifest_name="adni.csv",
    expected_columns=BASE_COLUMNS + BIOMARKER_COLUMNS,
)


AIBL = CohortDefinition(
    key="aibl",
    display_name="Australian Imaging Biomarkers and Lifestyle",
    release="current",
    role=CohortRole.CROSS_SITE,
    participants=768,
    modalities=("t1", "flair", "amyloid_pet"),
    diagnoses=("cn", "mci", "ad"),
    vendors=("siemens",),
    field_strengths=("1.5T", "3T"),
    access=AccessLevel.APPLICATION,
    manifest_name="aibl.csv",
    expected_columns=BASE_COLUMNS + BIOMARKER_COLUMNS,
)


NACC = CohortDefinition(
    key="nacc",
    display_name="NACC UDS v4",
    release="UDSv4",
    role=CohortRole.CROSS_SITE,
    participants=4200,
    modalities=("t1", "amyloid_pet", "clinical"),
    diagnoses=("cn", "mci", "ad"),
    vendors=("siemens", "ge", "philips"),
    field_strengths=("1.5T", "3T"),
    access=AccessLevel.APPLICATION,
    manifest_name="nacc.csv",
    expected_columns=BASE_COLUMNS + BIOMARKER_COLUMNS,
)


PPMI = CohortDefinition(
    key="ppmi",
    display_name="Parkinson Progression Marker Initiative 2.0",
    release="PPMI-2",
    role=CohortRole.CROSS_DISEASE,
    participants=1500,
    modalities=("t1", "flair", "swi", "datscan"),
    diagnoses=("healthy", "prodromal", "pd", "genetic_risk"),
    vendors=("siemens", "ge", "philips", "hitachi"),
    field_strengths=("1.5T", "3T"),
    access=AccessLevel.APPLICATION,
    manifest_name="ppmi.csv",
    expected_columns=BASE_COLUMNS + BIOMARKER_COLUMNS,
)


DIAN = CohortDefinition(
    key="dian",
    display_name="Dominantly Inherited Alzheimer Network Observational Study",
    release="current",
    role=CohortRole.PRESYMPTOMATIC,
    participants=214,
    modalities=("t1", "pib_pet", "clinical", "genetics"),
    diagnoses=("mutation_carrier", "non_carrier"),
    vendors=("siemens", "ge", "philips"),
    field_strengths=("3T",),
    access=AccessLevel.APPLICATION,
    manifest_name="dian.csv",
    expected_columns=BASE_COLUMNS + BIOMARKER_COLUMNS,
)


HCP_AGING = CohortDefinition(
    key="hcp_aging",
    display_name="HCP Lifespan Aging",
    release="Aging-2.0",
    role=CohortRole.REFERENCE,
    participants=725,
    modalities=("t1", "t2", "dwi", "rsfmri"),
    diagnoses=("healthy",),
    vendors=("siemens",),
    field_strengths=("3T",),
    access=AccessLevel.APPLICATION,
    manifest_name="hcp_aging.csv",
    expected_columns=BASE_COLUMNS + BIOMARKER_COLUMNS,
)


OPENNEURO = CohortDefinition(
    key="openneuro",
    display_name="OpenNeuro neurodegeneration collections",
    release="dataset-specific",
    role=CohortRole.ROBUSTNESS,
    participants=312,
    modalities=("t1",),
    diagnoses=("mixed",),
    vendors=("siemens", "ge", "philips", "canon"),
    field_strengths=("1.5T", "3T", "7T"),
    access=AccessLevel.OPEN,
    manifest_name="openneuro.csv",
    expected_columns=BASE_COLUMNS + BIOMARKER_COLUMNS,
)


COHORTS: Mapping[str, CohortDefinition] = {
    cohort.key: cohort
    for cohort in (
        UK_BIOBANK,
        ADNI,
        AIBL,
        NACC,
        PPMI,
        DIAN,
        HCP_AGING,
        OPENNEURO,
    )
}


LEFT_CORTICAL_REGIONS = (
    "bankssts",
    "caudalanteriorcingulate",
    "caudalmiddlefrontal",
    "cuneus",
    "entorhinal",
    "fusiform",
    "inferiorparietal",
    "inferiortemporal",
    "isthmuscingulate",
    "lateraloccipital",
    "lateralorbitofrontal",
    "lingual",
    "medialorbitofrontal",
    "middletemporal",
    "parahippocampal",
    "paracentral",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
    "pericalcarine",
    "postcentral",
    "posteriorcingulate",
    "precentral",
    "precuneus",
    "rostralanteriorcingulate",
    "rostralmiddlefrontal",
    "superiorfrontal",
    "superiorparietal",
    "superiortemporal",
    "supramarginal",
    "frontalpole",
    "temporalpole",
    "transversetemporal",
    "insula",
)


RIGHT_CORTICAL_REGIONS = tuple(f"right_{name}" for name in LEFT_CORTICAL_REGIONS)
ANATOMICAL_REGIONS = tuple(f"left_{name}" for name in LEFT_CORTICAL_REGIONS) + RIGHT_CORTICAL_REGIONS


def validate_cohort_manifest(cohort: CohortDefinition, header: Sequence[str]) -> tuple[str, ...]:
    available = set(header)
    return tuple(column for column in cohort.expected_columns if column not in available)


def locate_manifests(directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for key, cohort in COHORTS.items():
        candidate = directory / cohort.manifest_name
        if candidate.is_file():
            result[key] = candidate
    return result


def protocol_vocabulary() -> dict[tuple[str, str, str], int]:
    values: set[tuple[str, str, str]] = set()
    for cohort in COHORTS.values():
        for vendor in cohort.vendors:
            for field in cohort.field_strengths:
                for modality in cohort.modalities:
                    values.add((vendor, field, modality))
    return {value: index for index, value in enumerate(sorted(values))}


def cohorts_for_role(role: CohortRole) -> tuple[CohortDefinition, ...]:
    return tuple(cohort for cohort in COHORTS.values() if cohort.role == role)


def total_declared_participants() -> int:
    return sum(cohort.participants for cohort in COHORTS.values())
