"""
This is an example script to generate the outcome variable given the input dataset.

This script should be modified to prepare your own submission that predicts 
the outcome for the benchmark challenge by changing the clean_df and predict_outcomes function.

The predict_outcomes function takes a Pandas data frame. The return value must
be a data frame with two columns: nomem_encr and outcome. The nomem_encr column
should contain the nomem_encr column from the input data frame. The outcome
column should contain the predicted outcome for each nomem_encr. The outcome
should be 0 (no child) or 1 (having a child).

clean_df should be used to clean (preprocess) the data.

run.py can be used to test your submission.
"""

# List your libraries and modules here. Don't forget to update environment.yml!
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib


def clean_df(df, background_df=None):
    """
    Preprocess the input dataframe to feed the model.
    # If no cleaning is done (e.g. if all the cleaning is done in a pipeline) leave only the "return df" command

    Parameters:
    df (pd.DataFrame): The input dataframe containing the raw data (e.g., from PreFer_train_data.csv or PreFer_fake_data.csv).
    background (pd.DataFrame): Optional input dataframe containing background data (e.g., from PreFer_train_background_data.csv or PreFer_fake_background_data.csv).

    Returns:
    pd.DataFrame: The cleaned dataframe with only the necessary columns and processed variables.
    """

    ## This script contains a bare minimum working example
    # Create new variable with age
    df["age"] = 2024 - df["birthyear_bg"]

    # Imputing missing values in age with the mean
    df["age"] = df["age"].fillna(df["age"].mean())

    # Selecting variables for modelling
    keepcols = [
        "nomem_encr",  # ID variable required for predictions,
        "age",          # newly created variable
        "ci08a001", " ci09b001", " ci10c001", " ci11d001", " ci12e001", " ci13f001", " ci14g001", " ci15h001", " ci16i001", " ci17j001", " ci18k001", " ci19l001", " ci20m001",
        "ci08a002", " ci09b002", " ci10c002", " ci11d002", " ci12e002", " ci13f002", " ci14g002", " ci15h002", " ci16i002", " ci17j002", " ci18k002", " ci19l002", " ci20m002",
        "ci08a003", " ci09b003", " ci10c003", " ci11d003", " ci12e003", " ci13f003", " ci14g003", " ci15h003", " ci16i003", " ci17j003", " ci18k003", " ci19l003", " ci20m003",
        "ci08a004", " ci09b004", " ci10c004", " ci11d004", " ci12e004", " ci13f004", " ci14g004", " ci15h004", " ci16i004", " ci17j004", " ci18k004",
        "ci08a005", " ci09b005", " ci10c005", " ci11d005", " ci12e005", " ci13f005", " ci14g005", " ci15h005", " ci16i005", " ci17j005", " ci18k005", " ci19l005", " ci20m005",
        "ci08a006", " ci09b006", " ci10c006", " ci11d006", " ci12e006", " ci13f006", " ci14g006", " ci15h006", " ci16i006", " ci17j006", " ci18k006", " ci19l006", " ci20m006",
        "ci08a007", " ci09b007", " ci10c007", " ci11d007", " ci12e007", " ci13f007", " ci14g007", " ci15h007", " ci16i007", " ci17j007", " ci18k007", " ci19l007", " ci20m007",
        "ci08a008", " ci09b008", " ci10c008", " ci11d008", " ci12e008", " ci13f008", " ci14g008", " ci15h008", " ci16i008", " ci17j008", " ci18k008", " ci19l008", " ci20m008",
        "ci08a009", " ci09b009", " ci10c009", " ci11d009", " ci12e009", " ci13f009", " ci14g009", " ci15h009", " ci16i009", " ci17j009", " ci18k009", " ci19l009", " ci20m009",
        "ci08a010", " ci09b010", " ci10c010", " ci11d010", " ci12e010", " ci13f010", " ci14g010", " ci15h010", " ci16i010", " ci17j010", " ci18k010",
        "ci08a011", " ci09b011", " ci10c011", " ci11d011", " ci12e011", " ci13f011", " ci14g011", " ci15h011", " ci16i011", " ci17j011", " ci18k011",
        "ci08a012", " ci09b012", " ci10c012", " ci11d012", " ci12e012", " ci13f012", " ci14g012", " ci15h012", " ci16i012", " ci17j012", " ci18k012",
        "ci08a087", " ci09b087", " ci10c087", " ci11d087", " ci12e087", " ci13f087", " ci14g087", " ci15h087", " ci16i087", " ci17j087", " ci18k087", " ci19l087", " ci20m087",
        "ci08a088", " ci09b088", " ci10c088", " ci11d088", " ci12e088", " ci13f088", " ci14g088", " ci15h088", " ci16i088", " ci17j088", " ci18k088",
        "ci08a089", " ci09b089", " ci10c089", " ci11d089", " ci12e089", " ci13f089", " ci14g089", " ci15h089", " ci16i089", " ci17j089", " ci18k089",
        "ci08a090", " ci09b090", " ci10c090", " ci11d090", " ci12e090", " ci13f090", " ci14g090", " ci15h090", " ci16i090", " ci17j090", " ci18k090", " ci19l090", " ci20m090",
        "ci08a091", " ci09b091", " ci10c091", " ci11d091", " ci12e091", " ci13f091", " ci14g091", " ci15h091", " ci16i091", " ci17j091", " ci18k091", " ci19l091", " ci20m091",
        "ci08a092", " ci09b092", " ci10c092", " ci11d092", " ci12e092", " ci13f092", " ci14g092", " ci15h092", " ci16i092", " ci17j092", " ci18k092", " ci19l092", " ci20m092",
        "ci08a093", " ci09b093", " ci10c093",
        "ci08a094", " ci09b094", " ci10c094", " ci11d094", " ci12e094", " ci13f094", " ci14g094", " ci15h094", " ci16i094", " ci17j094", " ci18k094", " ci19l094", " ci20m094",
        "ci08a095", " ci09b095", " ci10c095", " ci11d095", " ci12e095", " ci13f095", " ci14g095", " ci15h095", " ci16i095", " ci17j095", " ci18k095", " ci19l095", " ci20m095",
        "ci08a096", " ci09b096", " ci10c096", " ci11d096", " ci12e096", " ci13f096", " ci14g096", " ci15h096", " ci16i096", " ci17j096", " ci18k096", " ci19l096", " ci20m096",
        "ci08a097", " ci09b097", " ci10c097", " ci11d097", " ci12e097",
        "ci08a098", " ci09b098", " ci10c098", " ci11d098", " ci12e098", " ci13f098", " ci14g098", " ci15h098", " ci16i098", " ci17j098", " ci18k098", " ci19l098", " ci20m098",
        "ci08a099", " ci09b099", " ci10c099", " ci11d099", " ci12e099", " ci13f099", " ci14g099", " ci15h099", " ci16i099", " ci17j099", " ci18k099", " ci19l099", " ci20m099",
        "ci08a100", " ci09b100", " ci10c100", " ci11d100", " ci12e100", " ci13f100", " ci14g100", " ci15h100", " ci16i100", " ci17j100", " ci18k100", " ci19l100", " ci20m100",
        "ci08a101", " ci09b101", " ci10c101", " ci11d101", " ci12e101", " ci13f101", " ci14g101", " ci15h101", " ci16i101", " ci17j101", " ci18k101", " ci19l101", " ci20m101",
        "ci08a243", " ci09b243", " ci10c243", " ci11d243", " ci12e243", " ci13f243", " ci14g243", " ci15h243", " ci16i243", " ci17j243", " ci18k243",
        "ci08a244", " ci09b244", " ci10c244", " ci11d244", " ci12e244", " ci13f244", " ci14g244", " ci15h244", " ci16i244", " ci17j244", " ci18k244",
        "ci08a245", " ci09b245", " ci10c245", " ci11d245", " ci12e245", " ci13f245", " ci14g245", " ci15h245", " ci16i245", " ci17j245", " ci18k245", " ci19l245", " ci20m245",
        "ci08a246", " ci09b246", " ci10c246", " ci11d246", " ci12e246", " ci13f246", " ci14g246", " ci15h246", " ci16i246", " ci17j246", " ci18k246", " ci19l246", " ci20m246",
        "ci08a247", " ci09b247", " ci10c247", " ci11d247", " ci12e247", " ci13f247", " ci14g247", " ci15h247", " ci16i247", " ci17j247", " ci18k247", " ci19l247", " ci20m247",
        "ci08a248", " ci09b248", " ci10c248", " ci11d248", " ci12e248", " ci13f248", " ci14g248", " ci15h248", " ci16i248", " ci17j248", " ci18k248", " ci19l248", " ci20m248",
        "ci08a249", " ci09b249", " ci10c249", " ci11d249", " ci12e249", " ci13f249", " ci14g249", " ci15h249", " ci16i249", " ci17j249", " ci18k249", " ci19l249", " ci20m249",
        "ci08a250", " ci09b250", " ci10c250", " ci11d250", " ci12e250", " ci13f250", " ci14g250", " ci15h250", " ci16i250", " ci17j250", " ci18k250", " ci19l250", " ci20m250",
        "ci08a251", " ci09b251", " ci10c251", " ci11d251", " ci12e251", " ci13f251", " ci14g251", " ci15h251", " ci16i251", " ci17j251", " ci18k251", " ci19l251", " ci20m251",
        "ci08a252", " ci09b252", " ci10c252", " ci11d252", " ci12e252", " ci13f252", " ci14g252", " ci15h252", " ci16i252", " ci17j252", " ci18k252", " ci19l252", " ci20m252",
        "ch08b001", " ch09c001", " ch10d001", " ch11e001", " ch12f001", " ch13g001", " ch15h001", " ch16i001", " ch17j001", " ch18k001", " ch19l001", " ch20m001",
        "ch08b002", " ch09c002", " ch10d002", " ch11e002", " ch12f002", " ch13g002", " ch15h002", " ch16i002", " ch17j002", " ch18k002", " ch19l002", " ch20m002",
        "ch08b003", " ch09c003", " ch10d003", " ch11e003", " ch12f003", " ch13g003", " ch15h003", " ch16i003", " ch17j003", " ch18k003", " ch19l003", " ch20m003",
        "ch07a011", " ch08b011", " ch09c011", " ch10d011", " ch11e011", " ch12f011", " ch13g011", " ch15h011", " ch16i011", " ch17j011", " ch18k011", " ch19l011", " ch20m011",
        "ch07a012", " ch08b012", " ch09c012", " ch10d012", " ch11e012", " ch12f012", " ch13g012", " ch15h012", " ch16i012", " ch17j012", " ch18k012", " ch19l012", " ch20m012",
        "ch07a013", " ch08b013", " ch09c013", " ch10d013", " ch11e013", " ch12f013", " ch13g013", " ch15h013", " ch16i013", " ch17j013", " ch18k013", " ch19l013", " ch20m013",
        "ch07a014", " ch08b014", " ch09c014", " ch10d014", " ch11e014", " ch12f014", " ch13g014", " ch15h014", " ch16i014", " ch17j014", " ch18k014", " ch19l014", " ch20m014",
        "ch07a015", " ch08b015", " ch09c015", " ch10d015", " ch11e015", " ch12f015", " ch13g015", " ch15h015", " ch16i015", " ch17j015", " ch18k015", " ch19l015", " ch20m015",
        "ch07a018", " ch08b018", " ch09c018", " ch10d018", " ch11e018", " ch12f018", " ch13g018", " ch15h018", " ch16i018", " ch17j018", " ch18k018", " ch19l018", " ch20m018",
        "ch07a020", " ch08b020", " ch09c020", " ch10d020", " ch11e020", " ch12f020", " ch13g020", " ch15h020", " ch16i020", " ch17j020", " ch18k020", " ch19l020", " ch20m020",
        "ch07a021", " ch08b021", " ch09c021", " ch10d021", " ch11e021", " ch12f021", " ch13g021", " ch15h021", " ch16i021", " ch17j021", " ch18k021", " ch19l021", " ch20m021",
        "ch07a022", " ch08b022", " ch09c022", " ch10d022", " ch11e022", " ch12f022", " ch13g022", " ch15h022", " ch16i022", " ch17j022", " ch18k022", " ch19l022", " ch20m022",
        "ch07a206", " ch08b206", " ch09c206", " ch10d206", " ch11e206", " ch12f206", " ch13g206", " ch15h206", " ch16i206", " ch17j206", " ch18k206", " ch19l206", " ch20m206",
        "vch07a207", " ch08b207", " ch09c207", " ch10d207", " ch11e207", " ch12f207", " ch13g207", " ch15h207", " ch16i207", " ch17j207", " ch18k207", " ch19l207", " ch20m207",
        "ch07a208", " ch08b208", " ch09c208", " ch10d208", " ch11e208", " ch12f208", " ch13g208", " ch15h208", " ch16i208", " ch17j208", " ch18k208", " ch19l208", " ch20m208",
        "ch07a209", " ch08b209", " ch09c209", " ch10d209", " ch11e209", " ch12f209", " ch13g209", " ch15h209", " ch16i209", " ch17j209", " ch18k209", " ch19l209", " ch20m209",
        "ch07a210", " ch08b210", " ch09c210", " ch10d210", " ch11e210", " ch12f210", " ch13g210", " ch15h210", " ch16i210", " ch17j210", " ch18k210", " ch19l210", " ch20m210",
        "ch07a211", " ch08b211", " ch09c211", " ch10d211", " ch11e211", " ch12f211", " ch13g211", " ch15h211", " ch16i211", " ch17j211", " ch18k211", " ch19l211", " ch20m211",
        "ch07a212", " ch08b212", " ch09c212", " ch10d212", " ch11e212", " ch12f212", " ch13g212", " ch15h212", " ch16i212", " ch17j212", " ch18k212", " ch19l212", " ch20m212",
        "ch07a213", " ch08b213", " ch09c213", " ch10d213", " ch11e213", " ch12f213", " ch13g213", " ch15h213", " ch16i213", " ch17j213", " ch18k213", " ch19l213", " ch20m213",
        "ch07a214", " ch08b214", " ch09c214", " ch10d214", " ch11e214", " ch12f214", " ch13g214", " ch15h214", " ch16i214", " ch17j214", " ch18k214", " ch19l214", " ch20m214",
        "ch07a215", " ch08b215", " ch09c215", " ch10d215", " ch11e215", " ch12f215", " ch13g215", " ch15h215", " ch16i215", " ch17j215", " ch18k215", " ch19l215", " ch20m215",
        "ch07a216", " ch08b216", " ch09c216", " ch10d216", " ch11e216", " ch12f216", " ch13g216", " ch15h216", " ch16i216", " ch17j216", " ch18k216", " ch19l216", " ch20m216",
        "ch07a217", " ch08b217", " ch09c217", " ch10d217", " ch11e217", " ch12f217", " ch13g217", " ch15h217", " ch16i217", " ch17j217", " ch18k217", " ch19l217", " ch20m217",
        "ch07a218", " ch08b218", " ch09c218", " ch10d218", " ch11e218", " ch12f218", " ch13g218", " ch15h218", " ch16i218", " ch17j218", " ch18k218", " ch19l218", " ch20m218",
        "ch07a219", " ch08b219", " ch09c219", " ch10d219", " ch11e219", " ch12f219", " ch13g219", " ch15h219", " ch16i219", " ch17j219", " ch18k219", " ch19l219", " ch20m219",
        "ch07a220", " ch08b220", " ch09c220", " ch10d220", " ch11e220", " ch12f220", " ch13g220", " ch15h220", " ch16i220", " ch17j220", " ch18k220", " ch19l220", " ch20m220",
        "ch07a221", " ch08b221", " ch09c221", " ch10d221", " ch11e221", " ch12f221", " ch13g221", " ch15h221", " ch16i221", " ch17j221", " ch18k221", " ch19l221", " ch20m221",
        "ch07a222", " ch08b222", " ch09c222", " ch10d222", " ch11e222", " ch12f222", " ch13g222", " ch15h222", " ch16i222", " ch17j222", " ch18k222", " ch19l222", " ch20m222",
        "ch07a223", " ch08b223", " ch09c223", " ch10d223", " ch11e223", " ch12f223", " ch13g223", " ch15h223", " ch16i223", " ch17j223", " ch18k223", " ch19l223", " ch20m223",
        "ch07a224", " ch08b224", " ch09c224", " ch10d224", " ch11e224", " ch12f224", " ch13g224", " ch15h224", " ch16i224", " ch17j224", " ch18k224", " ch19l224", " ch20m224",
        "ch07a225", " ch08b225", " ch09c225", " ch10d225", " ch11e225", " ch12f225", " ch13g225", " ch15h225", " ch16i225", " ch17j225", " ch18k225", " ch19l225", " ch20m225",
        "ch07a226", " ch08b226", " ch09c226", " ch10d226", " ch11e226", " ch12f226", " ch13g226", " ch15h226", " ch16i226", " ch17j226", " ch18k226", " ch19l226", " ch20m226",
        "ch07a227", " ch08b227", " ch09c227", " ch10d227", " ch11e227", " ch12f227", " ch13g227", " ch15h227", " ch16i227", " ch17j227", " ch18k227", " ch19l227", " ch20m227",
        "ch07a228", " ch08b228", " ch09c228", " ch10d228", " ch11e228", " ch12f228", " ch13g228", " ch15h228", " ch16i228", " ch17j228", " ch18k228", " ch19l228", " ch20m228",
        "ch07a229", " ch08b229", " ch09c229", " ch10d229", " ch11e229", " ch12f229", " ch13g229", " ch15h229", " ch16i229", " ch17j229", " ch18k229", " ch19l229", " ch20m229",
        "ch07a230", " ch08b230", " ch09c230", " ch10d230", " ch11e230", " ch12f230", " ch13g230", " ch15h230", " ch16i230", " ch17j230", " ch18k230", " ch19l230", " ch20m230",
        "ch07a231", " ch08b231", " ch09c231", " ch10d231", " ch11e231", " ch12f231", " ch13g231", " ch15h231", " ch16i231", " ch17j231", " ch18k231", " ch19l231", " ch20m231",
        "cs08a379", " cs09b379", " cs10c379", " cs11d379", " cs12e379", " cs13f379", " cs14g379", " cs15h379", " cs16i379", " cs17j379", " cs18k379", " cs19l379", " cs20m379",
        "cs08a380", " cs09b380", " cs10c380", " cs11d380", " cs12e380", " cs13f380", " cs14g380", " cs15h380", " cs16i380", " cs17j380", " cs18k380", " cs19l380",
        "cs08a381", " cs09b381", " cs10c381", " cs11d381", " cs12e381", " cs13f381", " cs14g381", " cs15h381", " cs16i381", " cs17j381", " cs18k381", " cs19l381", " cs20m381",
        "cs08a382", " cs09b382", " cs10c382", " cs11d382", " cs12e382", " cs13f382", " cs14g382", " cs15h382", " cs16i382", " cs17j382", " cs18k382", " cs19l382", " cs20m382",
        "cs08a383", " cs09b383", " cs10c383", " cs11d383", " cs12e383", " cs13f383", " cs14g383", " cs15h383", " cs16i383", " cs17j383", " cs18k383", " cs19l383", "cs20m383",
        "cs08a384", "cs09b384", "cs10c384", "cs11d384", "cs12e384", "cs13f384", "cs14g384", "cs15h384", "cs16i384", "cs17j384", "cs18k384", "cs19l384",
        "cs08a385", "cs09b385", "cs10c385", "cs11d385", "cs12e385", "cs13f385", "cs14g385", "cs15h385", "cs16i385", "cs17j385", "cs18k385", "cs19l385", "cs20m385",
        "cs08a386", "cs09b386", "cs10c386", "cs11d386", "cs12e386", "cs13f386", "cs14g386", "cs15h386", "cs16i386", "cs17j386", "cs18k386", "cs19l386", "cs20m386",
        "cs08a387", "cs09b387", "cs10c387", "cs11d387", "cs12e387", "cs13f387", "cs14g387", "cs15h387", "cs16i387", "cs17j387", "cs18k387", "cs19l387", "cs20m387",
        "cs08a388", "cs09b388", "cs10c388", "cs11d388", "cs12e388", "cs13f388", "cs14g388", "cs15h388", "cs16i388", "cs17j388", "cs18k388", "cs19l388", "cs20m388",
        "cs08a389", "cs09b389", "cs10c389", "cs11d389", "cs12e389", "cs13f389", "cs14g389",
        "cs08a390", "cs09b390", "cs10c390", "cs11d390", "cs12e390", "cs13f390", "cs14g390", "cs15h390", "cs16i390", "cs17j390", "cs18k390", "cs19l390", "cs20m390",
        "cs08a391", "cs09b391", "cs10c391", "cs11d391", "cs12e391", "cs13f391", "cs14g391", "cs15h391", "cs16i391", "cs17j391", "cs18k391", "cs19l391",
        "cs08a392", "cs09b392", "cs10c392", "cs11d392", "cs12e392", "cs13f392", "cs14g392", "cs15h392", "cs16i392", "cs17j392", "cs18k392", "cs19l392", "cs20m392",
        "cs08a393", "cs09b393", "cs10c393", "cs11d393", "cs12e393", "cs13f393", "cs14g393", "cs15h393", "cs16i393", "cs17j393", "cs18k393", "cs19l393", "cs20m393",
        "cs08a394", "cs09b394", "cs10c394", "cs11d394", "cs12e394", "cs13f394", "cs14g394", "cs15h394", "cs16i394", "cs17j394", "cs18k394", "cs19l394", "cs20m394",
        "cs08a395", "cs09b395", "cs10c395", "cs11d395", "cs12e395", "cs13f395", "cs14g395", "cs15h395", "cs16i395", "cs17j395", "cs18k395", "cs19l395", "cs20m395",
        "cs08a396", "cs09b396", "cs10c396", "cs11d396", "cs12e396", "cs13f396", "cs14g396", "cs15h396", "cs16i396", "cs17j396", "cs18k396", "cs19l396", "cs20m396",
        "cs08a397", "cs09b397", "cs10c397", "cs11d397", "cs12e397", "cs13f397", "cs14g397", "cs15h397", "cs16i397", "cs17j397", "cs18k397", "cs19l397", "cs20m397",
        "cs08a398", "cs09b398", "cs10c398", "cs11d398", "cs12e398", "cs13f398", "cs14g398", "cs15h398", "cs16i398", "cs17j398", "cs18k398", "cs19l398", "cs20m398",
        "cs08a399", "cs09b399", "cs10c399", "cs11d399", "cs12e399", "cs13f399", "cs14g399", "cs15h399", "cs16i399", "cs17j399", "cs18k399", "cs19l399", "cs20m399",
        "cs08a400", "cs09b400", "cs10c400", "cs11d400", "cs12e400", "cs13f400", "cs14g400", "cs15h400", "cs16i400", "cs17j400", "cs18k400", "cs19l400", "cs20m400",
        "cs08a401", "cs09b401", "cs10c401", "cs11d401", "cs12e401", "cs13f401", "cs14g401", "cs15h401", "cs16i401", "cs17j401", "cs18k401", "cs19l401", "cs20m401",
        "cs08a402", "cs09b402", "cs10c402", "cs11d402", "cs12e402", "cs13f402", "cs14g402", "cs15h402", "cs16i402", "cs17j402", "cs18k402", "cs19l402", "cs20m402",
        "cs08a403", "cs09b403", "cs10c403", "cs11d403", "cs12e403", "cs13f403", "cs14g403", "cs15h403", "cs16i403", "cs17j403", "cs18k403", "cs19l403", "cs20m403",
        "cs08a404", "cs09b404", "cs10c404", "cs11d404", "cs12e404", "cs13f404", "cs14g404", "cs15h404", "cs16i404", "cs17j404", "cs18k404", "cs19l404", "cs20m404",
        "cs08a405", "cs09b405", "cs10c405", "cs11d405", "cs12e405", "cs13f405", "cs14g405", "cs15h405", "cs16i405", "cs17j405", "cs18k405", "cs19l405", "cs20m405",
        "cs08a406", "cs09b406", "cs10c406", "cs11d406", "cs12e406", "cs13f406", "cs14g406", "cs15h406", "cs16i406", "cs17j406", "cs18k406", "cs19l406", "cs20m406",
        "cs08a407", "cs09b407", "cs10c407", "cs11d407", "cs12e407", "cs13f407", "cs14g407", "cs15h407", "cs16i407", "cs17j407", "cs18k407", "cs19l407", "cs20m407",
        "cs12e437", "cs13f437", "cs14g437", "cs15h437", "cs16i437", "cs17j437", "cs18k437", "cs19l437", "cs20m437",
        "cs12e438", "cs13f438", "cs14g438", "cs15h438", "cs16i438", "cs17j438", "cs18k438", "cs19l438", "cs20m438",
        "cs12e439", "cs13f439", "cs14g439", "cs15h439", "cs16i439", "cs17j439", "cs18k439", "cs19l439", "cs20m439",
        "cs12e440", "cs13f440", "cs14g440", "cs15h440", "cs16i440", "cs17j440", "cs18k440", "cs19l440", "cs20m440",
        "cs12e441", "cs13f441",
        "cs12e442", "cs13f442",
        "cs12e443", "cs13f443", "cs14g443", "cs15h443", "cs16i443", "cs17j443", "cs18k443", "cs19l443", "cs20m443",
        "cs14g472", "cs15h472", "cs16i472", "cs17j472", "cs18k472", "cs19l472", "cs20m472",
        "cs14g473", "cs15h473", "cs16i473", "cs17j473", "cs18k473", "cs19l473", "cs20m473",
        "cs14g474", "cs15h474", "cs16i474", "cs17j474", "cs18k474", "cs19l474", "cs20m474",
        "cs20m577",
        "cs20m578",
        "cs20m579",
        "cs20m580",
        "cs20m581",
        "cw08a004", "cw09b004", "cw10c004", "cw11d004", "cw12e004", "cw13f004", "cw14g004", "cw15h004", "cw16i004", "cw17j004", "cw18k004",
        "cw08a005", "cw09b005", "cw10c005", "cw11d005", "cw12e005", "cw13f005", "cw14g005", "cw15h005", "cw16i005", "cw17j005", "cw18k005", "cw20m005",
        "cw08a006", "cw09b006", "cw10c006", "cw11d006", "cw12e006", "cw13f006", "cw14g006", "cw15h006", "cw16i006", "cw17j006", "cw18k006", "cw19l006", "cw20m006",
        "cw08a007", "cw09b007", "cw10c007", "cw11d007", "cw12e007", "cw13f007", "cw14g007", "cw15h007", "cw16i007", "cw17j007", "cw18k007",
        "cw08a008", "cw09b008", "cw10c008", "cw11d008", "cw12e008", "cw13f008", "cw14g008", "cw15h008", "cw16i008", "cw17j008", "cw18k008", "cw19l008", "cw20m008",
        "cw08a009", "cw09b009", "cw10c009", "cw11d009", "cw12e009", "cw13f009", "cw14g009", "cw15h009", "cw16i009", "cw17j009", "cw18k009", "cw19l009", "cw20m009",
        "cw08a010", "cw09b010", "cw10c010", "cw11d010", "cw12e010", "cw13f010", "cw14g010", "cw15h010", "cw16i010", "cw17j010", "cw18k010",
        "cw08a011", "cw09b011", "cw10c011", "cw11d011", "cw12e011", "cw13f011", "cw14g011", "cw15h011", "cw16i011", "cw17j011", "cw18k011", "cw19l011", "cw20m011",
        "cw08a012", "cw09b012", "cw10c012", "cw11d012", "cw12e012", "cw13f012", "cw14g012", "cw15h012", "cw16i012", "cw17j012", "cw18k012", "cw19l012", "cw20m012",
        "cw08a013", "cw09b013", "cw10c013", "cw11d013", "cw12e013", "cw13f013", "cw14g013", "cw15h013", "cw16i013", "cw17j013", "cw18k013", "cw19l013", "cw20m013",
        "cw08a014", "cw09b014", "cw10c014", "cw11d014", "cw12e014", "cw13f014", "cw14g014", "cw15h014", "cw16i014", "cw17j014", "cw18k014", "cw19l014", "cw20m014",
        "cw08a015", "cw09b015", "cw10c015", "cw11d015", "cw12e015", "cw13f015", "cw14g015", "cw15h015", "cw16i015", "cw17j015", "cw18k015", "cw19l015", "cw20m015",
        "cw08a016", "cw09b016", "cw10c016", "cw11d016", "cw12e016", "cw13f016", "cw14g016", "cw15h016", "cw16i016", "cw17j016", "cw18k016", "cw19l016", "cw20m016",
        "cw08a017", "cw09b017", "cw10c017", "cw11d017", "cw12e017", "cw13f017", "cw14g017", "cw15h017", "cw16i017", "cw17j017", "cw18k017", "cw19l017", "cw20m017",
        "cw08a018", "cw09b018", "cw10c018", "cw11d018", "cw12e018", "cw13f018", "cw14g018", "cw15h018", "cw16i018", "cw17j018", "cw18k018", "cw19l018", "cw20m018",
        "cw08a019", "cw09b019", "cw10c019", "cw11d019", "cw12e019", "cw13f019", "cw14g019", "cw15h019", "cw16i019", "cw17j019", "cw18k019", "cw19l019", "cw20m019",
        "cw08a020", "cw09b020", "cw10c020", "cw11d020", "cw12e020", "cw13f020", "cw14g020", "cw15h020", "cw16i020", "cw17j020", "cw18k020", "cw19l020", "cw20m020",
        "cw08a021", "cw09b021", "cw10c021", "cw11d021", "cw12e021", "cw13f021", "cw14g021", "cw15h021", "cw16i021", "cw17j021", "cw18k021", "cw19l021", "cw20m021",
        "cw08a022", "cw09b022", "cw10c022", "cw11d022", "cw12e022", "cw13f022", "cw14g022", "cw15h022", "cw16i022", "cw17j022", "cw18k022", "cw19l022", "cw20m022",
        "cw08a023", "cw09b023", "cw10c023", "cw11d023", "cw12e023", "cw13f023", "cw14g023", "cw15h023", "cw16i023", "cw17j023", "cw18k023", "cw19l023", "cw20m023",
        "cw08a024", "cw09b024", "cw10c024", "cw11d024", "cw12e024", "cw13f024", "cw14g024", "cw15h024", "cw16i024", "cw17j024", "cw18k024", "cw19l024", "cw20m024",
        "cw08a025", "cw09b025", "cw10c025", "cw11d025", "cw12e025", "cw13f025", "cw14g025", "cw15h025", "cw16i025", "cw17j025", "cw18k025", "cw19l025", "cw20m025",
        "cw08a026", "cw09b026", "cw10c026", "cw11d026", "cw12e026", "cw13f026", "cw14g026", "cw15h026", "cw16i026", "cw17j026", "cw18k026", "cw19l026", "cw20m026",
        "cw08a027", "cw09b027", "cw10c027", "cw11d027", "cw12e027", "cw13f027", "cw14g027", "cw15h027", "cw16i027", "cw17j027", "cw18k027", "cw19l027", "cw20m027",
        "cw08a028", "cw09b028", "cw10c028", "cw11d028", "cw12e028", "cw13f028", "cw14g028", "cw15h028", "cw16i028", "cw17j028", "cw18k028", "cw19l028",
        "cw08a088", "cw09b088", "cw10c088", "cw11d088", "cw12e088", "cw13f088", "cw14g088", "cw15h088", "cw16i088", "cw17j088", "cw18k088", "cw19l088", "cw20m088",
        "cw08a089", "cw09b089", "cw10c089", "cw11d089", "cw12e089", "cw13f089", "cw14g089", "cw15h089", "cw16i089", "cw17j089", "cw18k089", "cw19l089", "cw20m089",
        "cw08a090", "cw09b090", "cw10c090", "cw11d090", "cw12e090", "cw13f090", "cw14g090", "cw15h090", "cw16i090", "cw17j090", "cw18k090", "cw19l090", "cw20m090",
        "cw08a091", "cw09b091", "cw10c091", "cw11d091", "cw12e091", "cw13f091", "cw14g091", "cw15h091", "cw16i091", "cw17j091", "cw18k091", "cw19l091", "cw20m091",
        "cw08a092", "cw09b092", "cw10c092", "cw11d092", "cw12e092", "cw13f092", "cw14g092", "cw15h092", "cw16i092", "cw17j092", "cw18k092", "cw19l092", "cw20m092",
        "cw08a093", "cw09b093", "cw10c093", "cw11d093", "cw12e093", "cw13f093", "cw14g093", "cw15h093", "cw16i093", "cw17j093", "cw18k093", "cw19l093", "cw20m093",
        "cw08a094", "cw09b094", "cw10c094", "cw11d094", "cw12e094", "cw13f094", "cw14g094", "cw15h094", "cw16i094", "cw17j094", "cw18k094", "cw19l094", "cw20m094",
        "cw08a095", "cw09b095", "cw10c095", "cw11d095", "cw12e095", "cw13f095", "cw14g095", "cw15h095", "cw16i095", "cw17j095", "cw18k095", "cw19l095", "cw20m095",
        "cw08a096", "cw09b096", "cw10c096", "cw11d096", "cw12e096", "cw13f096", "cw14g096", "cw15h096", "cw16i096", "cw17j096", "cw18k096", "cw19l096", "cw20m096",
        "cw08a097", "cw09b097", "cw10c097", "cw11d097", "cw12e097", "cw13f097", "cw14g097", "cw15h097", "cw16i097", "cw17j097", "cw18k097", "cw19l097", "cw20m097",
        "cw08a098", "cw09b098", "cw10c098", "cw11d098", "cw12e098", "cw13f098", "cw14g098", "cw15h098", "cw16i098", "cw17j098", "cw18k098", "cw19l098", "cw20m098",
        "cw08a099", "cw09b099", "cw10c099", "cw11d099", "cw12e099", "cw13f099", "cw14g099", "cw15h099", "cw16i099", "cw17j099", "cw18k099", "cw19l099", "cw20m099",
        "cw08a100", "cw09b100", "cw10c100", "cw11d100", "cw12e100", "cw13f100", "cw14g100", "cw15h100", "cw16i100", "cw17j100", "cw18k100", "cw19l100", "cw20m100",
        "cw08a101", "cw09b101", "cw10c101", "cw11d101", "cw12e101", "cw13f101", "cw14g101", "cw15h101", "cw16i101", "cw17j101", "cw18k101", "cw19l101", "cw20m101",
        "cw08a102", "cw09b102", "cw10c102", "cw11d102", "cw12e102", "cw13f102", "cw14g102", "cw15h102", "cw16i102", "cw17j102", "cw18k102", "cw19l102", "cw20m102",
        "cw08a342", "cw09b342", "cw10c342", "cw11d342", "cw12e342", "cw13f342", "cw14g342", "cw15h342", "cw16i342", "cw17j342", "cw18k342", "cw19l342", "cw20m342",
        "cw08a343", "cw09b343", "cw10c343", "cw11d343", "cw12e343", "cw13f343", "cw14g343", "cw15h343", "cw16i343", "cw17j343", "cw18k343", "cw19l343", "cw20m343",
        "cw08a344", "cw09b344", "cw10c344", "cw11d344", "cw12e344", "cw13f344", "cw14g344", "cw15h344", "cw16i344", "cw17j344", "cw18k344", "cw19l344", "cw20m344",
        "cw08a345", "cw09b345", "cw10c345", "cw11d345", "cw12e345", "cw13f345", "cw14g345", "cw15h345", "cw16i345", "cw17j345", "cw18k345", "cw19l345", "cw20m345",
        "cw08a346", "cw09b346", "cw10c346", "cw11d346", "cw12e346", "cw13f346", "cw14g346", "cw15h346", "cw16i346", "cw17j346", "cw18k346", "cw19l346", "cw20m346",
        "cw08a347", "cw09b347", "cw10c347", "cw11d347", "cw12e347", "cw13f347", "cw14g347", "cw15h347", "cw16i347", "cw17j347", "cw18k347", "cw19l347", "cw20m347",
        "cw08a348", "cw09b348", "cw10c348", "cw11d348", "cw12e348", "cw13f348", "cw14g348", "cw15h348", "cw16i348", "cw17j348", "cw18k348", "cw19l348", "cw20m348",
        "cw08a349", "cw09b349", "cw10c349", "cw11d349", "cw12e349", "cw13f349", "cw14g349", "cw15h349", "cw16i349", "cw17j349", "cw18k349", "cw19l349", "cw20m349",
        "cw08a350", "cw09b350", "cw10c350", "cw11d350", "cw12e350", "cw13f350", "cw14g350", "cw15h350", "cw16i350", "cw17j350", "cw18k350", "cw19l350", "cw20m350",
        "cw08a351", "cw09b351", "cw10c351", "cw11d351", "cw12e351", "cw13f351", "cw14g351", "cw15h351", "cw16i351", "cw17j351", "cw18k351", "cw19l351", "cw20m351",
        "cw08a352", "cw09b352", "cw10c352", "cw11d352", "cw12e352", "cw13f352", "cw14g352", "cw15h352", "cw16i352", "cw17j352", "cw18k352", "cw19l352", "cw20m352",
        "cw16i540",
        "cw16i541",
        "cw16i542",
        "cw16i543",
        "cw16i544",
        "cw16i545",
        "cw16i546",
        "cw16i547",
        "cw16i548",
        "cw19l549", "cw20m549",
        "cw19l550", "cw20m550",
        "cw19l551", "cw20m551",
        "cw19l552", "cw20m552",
        "cw19l553", "cw20m553",
        "cw19l554", "cw20m554",
        "cw19l555", "cw20m555",
        "cw19l556",
        "cw19l557", "cw20m557",
        "cw19l558", "cw20m558",
        "cw19l559", "cw20m559",
        "cw19l560", "cw20m560",
        "cw19l561", "cw20m561",
        "cw19l562", "cw20m562",
        "cw19l563", "cw20m563",
        "cw19l564", "cw20m564",
        "cw19l565", "cw20m565",
        "cw19l566", "cw20m566",
        "cw19l567", "cw20m567",
        "cw19l568", "cw20m568",
        "cw19l569", "cw20m569",
        "cw19l570", "cw20m570",
        "cw19l582", "cw20m582",
        "cw19l583", "cw20m583",
        "cw19l584", "cw20m584",
        "cw19l585", "cw20m585",
        "cw19l586", "cw20m586",
        "cw19l587", "cw20m587",
        "cw19l588", "cw20m588",
        "cw19l589", "cw20m589",
        "cw19l590", "cw20m590",
        "cw19l591", "cw20m591",
        "cw19l592", "cw20m592",
        "cw19l593", "cw20m593",
        "cw19l594", "cw20m594",
        "cw19l595", "cw20m595",
        "cw19l596", "cw20m596",
        "cw19l597", "cw20m597"
    ] 

    # Keeping data with variables selected
    df = df[keepcols]

    return df


def predict_outcomes(df, background_df=None, model_path="model.joblib"):
    """Generate predictions using the saved model and the input dataframe.

    The predict_outcomes function accepts a Pandas DataFrame as an argument
    and returns a new DataFrame with two columns: nomem_encr and
    prediction. The nomem_encr column in the new DataFrame replicates the
    corresponding column from the input DataFrame. The prediction
    column contains predictions for each corresponding nomem_encr. Each
    prediction is represented as a binary value: '0' indicates that the
    individual did not have a child during 2021-2023, while '1' implies that
    they did.

    Parameters:
    df (pd.DataFrame): The input dataframe for which predictions are to be made.
    background_df (pd.DataFrame): The background dataframe for which predictions are to be made.
    model_path (str): The path to the saved model file (which is the output of training.py).

    Returns:
    pd.DataFrame: A dataframe containing the identifiers and their corresponding predictions.
    """

    ## This script contains a bare minimum working example
    if "nomem_encr" not in df.columns:
        print("The identifier variable 'nomem_encr' should be in the dataset")

    # Load the model
    model = joblib.load(model_path)

    # Preprocess the fake / holdout data
    df = clean_df(df, background_df)

    # Exclude the variable nomem_encr if this variable is NOT in your model
    vars_without_id = df.columns[df.columns != 'nomem_encr']

    # Generate predictions from model, should be 0 (no child) or 1 (had child)
    predictions = model.predict(df[vars_without_id])

    # Output file should be DataFrame with two columns, nomem_encr and predictions
    df_predict = pd.DataFrame(
        {"nomem_encr": df["nomem_encr"], "prediction": predictions}
    )

    # Return only dataset with predictions and identifier
    return df_predict
