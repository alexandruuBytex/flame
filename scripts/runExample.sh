set -x;

source $HOME/.bashrc

exampleName=$1
datasetFileName="dataset"

echo "Starting example: ${exampleName}"

if(($#!=1));
then
    echo "Please provide an example name!"
    exit 1
fi

if [[ $exampleName == "hier_mnist" ]];
then
    datasetFileName="dataset_na_us"
fi

if [[ $exampleName == "medmnist" ]];
then
    datasetFileName="dataset1"
fi

echo "Dataset: ${datasetFileName}"

pathPrefix="../examples/${exampleName}/"

echo "${pathPrefix}schema.json"

flamectl create design "${exampleName}" -d "${exampleName} example" --insecure
flamectl create schema "${pathPrefix}schema.json" --design "${exampleName}" --insecure
flamectl create code "${pathPrefix}${exampleName}.zip" --design "${exampleName}" --insecure

unzip -o "${pathPrefix}${exampleName}.zip"

datasetData=$(flamectl create dataset "${pathPrefix}${datasetFileName}.json" --insecure)
echo "datasetData: ${datasetData}"

datasetID=$( echo "${datasetData}" | grep -o -e "\".*\"" | tr -d "\"")
echo "datasetID: ${datasetID}"

cp "${pathPrefix}dataSpec.json" "${pathPrefix}tmp.json"

echo "$(jq --arg datasetID "$datasetID" '.fromSystem."0" = [ $datasetID ]' "${pathPrefix}tmp.json")" > "${pathPrefix}result.json"

cp "${pathPrefix}result.json" "${pathPrefix}dataSpec.json"
rm "${pathPrefix}tmp.json"
rm "${pathPrefix}result.json"

#echo "$(jq --arg jobID "test" '.fromSystem."0" = [ $jobID ]' "${pathPrefix}tmp.json")" > result.json
jobID=$( flamectl create job "${pathPrefix}job-new.json" --insecure | grep "ID: " | tr -d "ID: " )
echo "jobID: ${jobID}"

flamectl start job $jobID --insecure
flamectl get jobs --insecure

set +x;