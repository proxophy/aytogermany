import pandas as pd

if __name__ == "__main__":
    canidatesdf = pd.read_excel(
        "excelsheets/normalo2020.xlsx", sheet_name="Candidates")

    nightsdf = pd.read_excel(
        "excelsheets/normalo2020.xlsx", sheet_name="Nights", header=None)
    nightsdf = nightsdf.transpose()
    nightsdf.columns = nightsdf.iloc[0]
    nightsdf.drop(index=0, axis=0, inplace=True)

    mbdf = pd.read_excel(
        "excelsheets/normalo2020.xlsx", sheet_name="Matchboxes")
    
    headerl = list(nightsdf)

    nights = [(list(zip(headerl, night[:-1])), night[-1]) 
              for night in nightsdf.values.tolist()]
    print(len(nights))
    # for pairs, lights in nights:
    #     print(len(pairs))
