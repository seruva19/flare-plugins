import json
import guidance
from guidance import models, gen, block


def register():
    flare_phi2 = FlarePhi2()
    return {
        "flare-phi2": {
            "prompt-to-op": flare_phi2.text_to_task,
            "modify-prompt": flare_phi2.modify_prompt,
        }
    }


class FlarePhi2:
    phi2: models.Transformers = None

    def __init__(self):
        pass

    def load_phi2(self):
        if self.phi2 is None:
            self.phi2 = models.Transformers(
                "microsoft/phi-2",
                cache_dir="weights",
                resume_download=True,
                trust_remote_code=True,
                device="cuda",
            )

    def text_to_task(self, params, response):
        self.load_phi2()

        prompt = params["prompt"]
        lm = self.phi2 + extract_task(prompt)

        json_task = lm["flare_json"]
        response["json_task"] = json.loads(json_task)

        if params["settings"]["offload_models"]:
            print("flare: offloading phi-2")
            params["tools"].unload_model(self.phi2.model_obj)
            self.phi2 = None

    def modify_prompt(self, params, response):
        self.load_phi2()

        modified_prompt = params["last_prompt"]
        task = params["task"]

        lm = self.phi2 + create_new_prompt(
            modified_prompt,
            task.task,
            task.source,
            task.target,
        )

        json_prompt = lm["flare_prompt"]
        json_prompt_obj = json.loads(json_prompt)
        response["outPrompt"] = json_prompt_obj["output"]

        self.offload_if_required(params)

    def offload_if_required(self, params):
        if params["settings"]["offload_models"]:
            print("flare: offloading phi-2")
            params["tools"].unload_model(self.phi2.model_obj)
            self.phi2 = None


@guidance
def extract_task(lm, prompt):
    lm += f"""
            The answer should be presented in JSON format and contain fields "task", "source", "target", "size", "factor".
            Field "task" may take only these values: "draw", "remove", "replace", "undo", "retry", "upscale", "resize".
            When constructing response, think step-by-step. Be very precise, do not miss anything. You job is to stay as close to user words as possible.
            Carefully analyse user request and generate output by following these instructions.
            - Does user want to remove or delete some object? If the answer is positive, set "task" to "remove", set "source" to the exact words describing object to remove, then stop generating.
            - Does user want to replace some object? If the answer is positive, set "task" to "replace", set "source" to the exact words describing object to replace, set "target" to the exact words describing object to replace with, then stop generating.
            - Does user only wants to cancel? If the answer is positive, set "task" to "undo", then stop generating.
            - Does user only wants to upscale image? If the answer is positive, set "task" to "upscale", set "factor" to number by which user wants to upscale, then stop generating.
            - Does user only want change size of the image? If the answer is positive, set "task" to "resize", set "size" to comma-separated width and height provided by user or "1024,1024" if no value provided by the user, then stop generating.
            - Does user want to draw some object? If the answer is positive, set "task" to "draw", set "source" to the exact phrase user used to describe the scene to draw, set "size" to comma-separated width and height provided by user or "1024,1024" if no value provided by the user, then stop generating.
            The user request is {prompt}. The following is your response in JSON format:
    """

    with block("flare_json"):
        lm += f"""
        {{
            "task": "{gen('task', stop='"')}",
            "source": "{gen('source', stop='"')}",
            "target": "{gen('target', stop='"')}",
            "size": "{gen('size', stop='"')}",
            "factor": "{gen('factor', stop='"')}"
        }}"""

    return lm


@guidance
def create_new_prompt(lm, prompt, task, source, target):
    lm += (
        """The answer should be presented in JSON format and contain field "output"."""
    )
    if task == "draw":
        lm += f"""If you add the following description "{source}" into the description of the following scene "{prompt}", what would be the updated text in one sentence? The following is your response in JSON format: """

    if task == "remove":
        lm += f"""If you modify a text describing scene "{prompt}" so it would not include mentioning of "{source}", but without adding new context, words or sentences to the scene, only removing "{source}", what would be the new text? The following is your response in JSON format: """

    elif task == "replace":
        lm += f"""If you modify a text describing some scene "{prompt}" so any mentionings of "{source}" in it would be replaced by "{target}", but leave any other words before and after these mentionings intact, what would be text of the modified description? The following is your response in JSON format: """

    with block("flare_prompt"):
        lm += f"""
        {{
            "output": "{gen('output', stop='"')}"
        }}"""

    return lm
