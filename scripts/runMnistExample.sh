pathPrefix="../examples/mnist/"

echo "${pathPrefix}schema.json"

flamectl create design mnist -d "mnist example" --insecure
flamectl create schema "${pathPrefix}schema.json" --design mnist --insecure
flamectl create code "${pathPrefix}mnist.zip" --design mnist --insecure

unzip -o "${pathPrefix}mnist.zip"

jobID=$(flamectl create dataset "${pathPrefix}dataset.json" --insecure | grep -o -e "\".*\"" | tr -d "\"")

cp "${pathPrefix}dataSpec.json" "${pathPrefix}tmp.json"

echo "$(jq --arg jobID "$jobID" '.fromSystem."0" = [ $jobID ]' "${pathPrefix}tmp.json")" > "${pathPrefix}result.json"

cp "${pathPrefix}result.json" "${pathPrefix}dataSpec.json"
rm "${pathPrefix}tmp.json"
rm "${pathPrefix}result.json"

#echo "$(jq --arg jobID "test" '.fromSystem."0" = [ $jobID ]' "${pathPrefix}tmp.json")" > result.json
flamectl create job "${pathPrefix}job-new.json" --insecure
flamectl start job $jobID --insecure
flamectl get jobs --insecure
