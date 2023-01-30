pathPrefix="../examples/mnist/"

echo "${pathPrefix}schema.json"

flamectl create design mnist -d "mnist example" --insecure
flamectl create schema "${pathPrefix}schema.json" --design mnist --insecure
flamectl create code "${pathPrefix}mnist.zip" --design mnist --insecure

unzip -o "${pathPrefix}mnist.zip"

jobID=flamectl create dataset "${pathPrefix}dataset.json" --insecure | grep -o -e "\".*\""

cp "${pathPrefix}job.json" "${pathPrefix}tmp.json"

echo "$(jq --arg jobID "$jobID" '.dataSpec.fromSystem = $jobID' "${pathPrefix}tmp.json")" > result.json

#echo "$(jq '.dataSpec.fromSystem = "${jobID}"' testJq.json)" > result.json
flamectl create job "${pathPrefix}tmp.json" --insecure
flamectl start job $jobID --insecure
flamectl get jobs --insecure
