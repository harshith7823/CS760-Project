{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "PCA.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyOZgYcbv9Ivuq+7YeYWA5yi",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/harshith7823/CS760-Project/blob/clean_clinical_data/clean_clinical_data.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "id": "dzDKo4obCOiz"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import torch\n",
        "import numpy as np\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "import torch.optim as optim\n",
        "from torchvision import transforms\n",
        "from torch.utils.data import Dataset, DataLoader,IterableDataset\n",
        "from sklearn.model_selection import train_test_split"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def clean_ct_data(oppScrData):\n",
        "    # Delete rows with empty values\n",
        "    ct_data= oppScrData[[\"L1_HU_BMD\", \"TAT Area (cm2)\", 'Total Body                Area EA (cm2)',\n",
        "       'VAT Area (cm2)', 'SAT Area (cm2)', 'VAT/SAT     Ratio', 'Muscle HU',\n",
        "       ' Muscle Area (cm2)', 'L3 SMI (cm2/m2)', 'AoCa        Agatston',\n",
        "       'Liver HU    (Median)', 'Age at CT']]\n",
        "    n = ct_data.shape[0]\n",
        "    preprocessed_ct_data = []\n",
        "    for i in range(n):\n",
        "        row = ct_data.loc[i]\n",
        "        ignore = False\n",
        "        for j in row:\n",
        "          if pd.isna(j) or j == ' ': # There is an empty string somewhere in Liver column\n",
        "            ignore = True\n",
        "            break\n",
        "        if not ignore:\n",
        "          preprocessed_ct_data.append(row)\n",
        "    return np.array(preprocessed_ct_data, dtype=np.float32)\n"
      ],
      "metadata": {
        "id": "HEznW1BHD8Zj"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from torch.utils.data.dataloader import T\n",
        "def preprocess_clinical_data(oppScrData, mean=True):\n",
        "    clinical_data = oppScrData.filter(['BMI','BMI >30', 'Sex', 'Tobacco', 'Met Sx', 'FRAX 10y Fx Prob (Orange-w/ DXA)',\n",
        "                                'FRAX 10y Hip Fx Prob (Orange-w/ DXA)','FRS 10-year risk (%)' ], axis=1)\n",
        "    # Replace all _,X,blanks with nan\n",
        "    clinical_data = clinical_data.replace(r'_', np.nan, regex=True)\n",
        "    clinical_data = clinical_data.replace(r'X', np.nan, regex=True)\n",
        "    clinical_data = clinical_data.replace(r'^\\s*$', np.nan, regex=True)\n",
        "\n",
        "    # Fill na in bmi column with mean\n",
        "    clinical_data['BMI'].fillna(value=clinical_data['BMI'].mean(skipna=True), inplace=True)\n",
        "\n",
        "    # Fill na in bmi>30 column based on bmi col\n",
        "    clinical_data.loc[clinical_data.BMI>30, 'BMI >30'] = 1\n",
        "    clinical_data.loc[clinical_data.BMI<=30, 'BMI >30'] = -1\n",
        "    \n",
        "    clinical_data['Sex'] = np.where(clinical_data['Sex']=='Male',1,-1)\n",
        "    clinical_data['Met Sx'] = np.where(clinical_data['Met Sx']=='Y',1,-1) \n",
        "\n",
        "    # Treat no data in tobacco as no tobacco usage \n",
        "    clinical_data['Tobacco'] = np.where(clinical_data['Tobacco']=='Yes',1,-1) \n",
        "\n",
        "    clinical_data['FRS 10-year risk (%)'] = clinical_data['FRS 10-year risk (%)'].replace(\"<1\", 0.01, regex=True)\n",
        "    clinical_data['FRS 10-year risk (%)'] = clinical_data['FRS 10-year risk (%)'].replace(\">30\", 0.30, regex=True)\n",
        "    clinical_data['FRS 10-year risk (%)'] =  clinical_data['FRS 10-year risk (%)'] * 100\n",
        " \n",
        "    cols_to_be_filled = ['FRAX 10y Fx Prob (Orange-w/ DXA)','FRAX 10y Hip Fx Prob (Orange-w/ DXA)','FRS 10-year risk (%)']\n",
        "    for c in cols_to_be_filled:\n",
        "      print(c)\n",
        "      if mean:  \n",
        "        clinical_data[c].fillna(value=clinical_data[c].mean(skipna=True), inplace=True)\n",
        "      else :\n",
        "        clinical_data[c].fillna(value=clinical_data[c].median(skipna=True), inplace=True)\n",
        "\n",
        "    return np.array(clinical_data, dtype=np.float32)\n",
        "    # return clinical_data\n",
        "\n",
        "    "
      ],
      "metadata": {
        "id": "z-dsvc_gIU_c"
      },
      "execution_count": 119,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def normalize_data(data):\n",
        "    n = data.shape[1]    \n",
        "    for i in range(n-1):\n",
        "      print(i)\n",
        "      data[:,i] = (data[:,i] - np.min(data[:,i]))/(np.max(data[:,i])- np.min(data[:,i]))\n",
        "    return data"
      ],
      "metadata": {
        "id": "tLPIjSdgEAeY"
      },
      "execution_count": 116,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "oppScrData = pd.read_excel (r'sample_data/OppScrData.xlsx')  \n",
        "ct_data = clean_ct_data(oppScrData)\n",
        "ct_data= normalize_data(ct_data)"
      ],
      "metadata": {
        "id": "LxJFEQzGEQ9A"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "clinical_data = preprocess_clinical_data(oppScrData)\n",
        "clinical_data = normalize_data(clinical_data)\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_UPoG8jgJEy7",
        "outputId": "265b0bb4-c357-4838-efbf-3f7479f9f65a"
      },
      "execution_count": 120,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "FRAX 10y Fx Prob (Orange-w/ DXA)\n",
            "FRAX 10y Hip Fx Prob (Orange-w/ DXA)\n",
            "FRS 10-year risk (%)\n",
            "0\n",
            "1\n",
            "2\n",
            "3\n",
            "4\n",
            "5\n",
            "6\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "eHmSHP1CU_f5"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}