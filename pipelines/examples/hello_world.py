"""
Hello World Pipeline - Basic Kubeflow Pipeline Example
"""
from kfp import dsl
from kfp import compiler


@dsl.component(base_image="python:3.11-slim")
def say_hello(name: str) -> str:
    """Simple component that returns a greeting."""
    message = f"Hello, {name}! Welcome to Kubeflow Pipelines."
    print(message)
    return message


@dsl.component(base_image="python:3.11-slim")
def process_greeting(greeting: str) -> str:
    """Process the greeting message."""
    processed = greeting.upper()
    print(f"Processed: {processed}")
    return processed


@dsl.pipeline(
    name="hello-world-pipeline",
    description="A simple hello world pipeline to test Kubeflow setup"
)
def hello_world_pipeline(name: str = "Kubeflow User"):
    """
    Simple pipeline that:
    1. Generates a greeting
    2. Processes it
    """
    hello_task = say_hello(name=name)
    process_task = process_greeting(greeting=hello_task.output)


if __name__ == "__main__":
    # Compile the pipeline
    compiler.Compiler().compile(
        pipeline_func=hello_world_pipeline,
        package_path="hello_world_pipeline.yaml"
    )
    print("Pipeline compiled to hello_world_pipeline.yaml")
