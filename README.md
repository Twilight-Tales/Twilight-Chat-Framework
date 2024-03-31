# Twilight Chat

<p align="center">
<img src="public/logo_light.png" alt="Twilight Chat Logo" width="256"/>
</p>
Twilight Chat is a virtual book club host that helps elderly to engage in reading activities, in order to help 
mediate cognitive declines. 

## Concept

As chatbots evolve, ...

## Features

- **Flexible Integration**: Develop chatbots for various front-end delivery mechanisms, integration with Twilio and
  other system.
- **Database Integration**: Robust support for databases to store and retrieve essential chatbot data.
- **Language Model Integration**: Seamless integration with third-party LLMs like OpenAI or self-hosted models.
- **API Deployment**: Deploy your chatbot logic as a scalable API, ready to integrate with various front-end channels.
- **Developer-Centric Tools**: Comprehensive tools, from conversation flow design to testing and deployment.

## Getting Started

<!-- This section should contain installation instructions, basic setup, and a "hello world" example. -->

1. Clone the repository:

```bash
git clone https://github.com/Twilight-Tales/Twilight-Chat.git
```

2. Set up your environment:

Copy `example.env` into `.env` and change the values.

3. Docker:

If you are running docker and vLLM on the same machine, you need to set

```bash
VLLM_URL=http://host.docker.internal:8000/v1
``` 

instead of `localhost` for docker to access vllm on the host machine.

To just run the app:

```bash
docker build -t twilight .
docker run --env-file ./.env -p 1680:1680 -d twilight
```

Hot reload during development:

```bash
docker run -v $(pwd):/app --env-file ./.env -p 1680:1680 --add-host=host.docker.internal:host-gateway twilight \
python -m chainlit run /app/app.py -h --port 1680 -w
```

4. Run in pure python env:
   Create and activate your python virtual env:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run chainlit chatbot:

```bash
chainlit run app.py -w
```

5. Deploy on k8s:

Create k8s secrets from `.env` file:

```bash
kubectl create secret generic twilight-secret --from-env-file=.env
```

push image to registry:

```bash
docker tag twilight quay.io/twilight_chat/twilight_chat
docker push quay.io/twilight_chat/twilight_chat
```

run this will create both app and service

```bash
kubectl apply -f twilight-app-k8s.yaml
```

port forward to host:

```bash
kubectl port-forward service/twilight-service 1680:1680
```

port forward in background:

```bash
kubectl port-forward service/twilight-service 1680:1680 > /dev/null 2>&1 &
```

confirm port forwarding works:
```bash
ps aux | grep 'kubectl port-forward'
```

6. Some of the commands above are included in the Makefile for your convenience. For example:

Build the docker image:
```bash
make build
```

Build and push image, update k8s deployment and restart port-forwarding:
```bash
make update-app
```

Checkout more short-cut commands in the Makefile.

## Contributing

We welcome contributions! Whether it's bug reports, feature requests, or pull requests, we encourage you to share and
contribute to Twilight's growth. Please read our CONTRIBUTING.md for guidelines on how to contribute.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE.md file for details.

## Acknowledgements

Thanks to all our contributors and supporters.

