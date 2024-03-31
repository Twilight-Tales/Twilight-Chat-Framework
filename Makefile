# Variables
include .env
export

DOCKER_IMAGE := $(REGISTRY)/$(NAMESPACE)/$(APP_NAME)

run-local:
	chainlit run app.py -w --port $(APP_PORT)

# Docker commands
build:
	docker build -t $(DOCKER_IMAGE) .

push:
	docker push $(DOCKER_IMAGE)

run:
	docker run --env-file .env -p $(APP_PORT):$(APP_PORT) -d --name $(APP_NAME) $(DOCKER_IMAGE)

stop:
	docker stop $(APP_NAME)

remove: stop
	docker rm $(APP_NAME)

# k8s commands
create-secret:
	kubectl create secret generic twilight-secret --from-env-file=.env

update-secret:
	kubectl delete secret twilight-secret
	kubectl create secret generic twilight-secret --from-env-file=.env

create-app:
	kubectl apply -f twilight-app-k8s.yaml

update-app: build push
	kubectl rollout restart deployment twilight-deployment
	kubectl rollout status deployment/twilight-deployment --timeout=120s
	sleep 5
	kubectl port-forward service/twilight-service $(APP_PORT):$(APP_PORT) > /dev/null 2>&1 &

# Clean up artifacts
clean:
	find . \( -name "__pycache__" -o -name ".pytest_cache" -o -name "results.db" -o -name ".coverage" \) -exec rm -rf {} +