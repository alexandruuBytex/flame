minikube stop
#minikube delete

minikube start --driver=hyperkit --cpus=6 --memory=6g --disk-size 100gb
minikube addons enable ingress
minikube addons enable ingress-dns

../fiab/setup-cert-manager.sh
../fiab/build-image.sh
eval $(minikube docker-env)