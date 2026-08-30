# PathMap Agent

PathMap Agent generates drug-associated Reactome pathways and gene sets from a drug name or SMILES string.

## Run PathMap

Run the following commands from the repository root.

### 1. Download the required resources

Download `CLADD.zip` and `local_data.zip` from the [Zenodo record](https://doi.org/10.5281/zenodo.22170086), place both files in the repository root, and extract them:

```bash
unzip -o CLADD.zip
unzip -o local_data.zip
```

After extraction, the repository root must contain:

```text
CLADD/
local_data/
```

### 2. Create the environment

```bash
conda env create -f PathMap.yml
conda activate MACL
```

### 3. Add your API keys

Add your own API keys in the following local files:

- Put your OpenAI API key in `openai_api_key.txt` and `GeneAgent/openai_GeneAgent_key`.
- Set `OPENAI_API_KEY=your_openai_key` in `CLADD/.env`.
- Put your Anthropic API key in `claude_api_key.txt`.

### 4. Run the workflow

Run the four steps in order. This example uses Gefitinib:

```bash
python Step1_Initial_description.py --drug-name Gefitinib
python Step2_choose_pathway_updated_openai_claude.py
python Step3_getgeneset_rule_fused.py
python Step4_Check_for_deficiencies.py
```

The final reviewed gene set is written to `result3/drug_geneset_checked/`.

### 5. Run downstream analyses (optional)

After generating the gene set in Steps 1–4, download `DownStream Task.zip` from the [Zenodo record](https://doi.org/10.5281/zenodo.22170086), place it in the repository root, and extract it:

```bash
unzip -o "DownStream Task.zip"
```

The extracted files support Step 5 of the PathMap workflow: downstream drug-response tasks based on the drug-specific gene set generated in Steps 1–4. Run the desired downstream task by following the methods in its original paper and the instructions in its official GitHub repository:

- [BANDRP](https://github.com/heckletbot/BANDRP)
- [CellHit](https://github.com/raimondilab/CellHit)
- [TransDRP](https://github.com/liuxuan666/TransDRP)
