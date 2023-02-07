minikube stop
minikube delete

#minikube start --driver=hyperkit --cpus=6 --memory=6g --disk-size 100gb
minikube start --driver=hyperkit --cpus=6 --memory=6g --disk-size 100gb --kubernetes-version=v1.23.8
#minikube service flame-mlflow --url -n flame

minikube addons enable ingress
minikube addons enable ingress-dns

eval $(minikube docker-env)

../fiab/setup-cert-manager.sh
../fiab/build-image.sh