import openai
import os
import time
import json
import subprocess
import base64

class TypeChatResult():
    def __init__(self):
        self.success = None
        self.message = "Result not initialized"
        self.data = {}
    
    def __str__(self):
        if self.success is None:
            return "Invalid result object"
        elif self.success:
            return "Success. " + str(self.data)
        elif not self.success:
            return "Error: " + self.message
    
    def to_dict(self):
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data
        }

    def from_dict(self, data):
        self.success = data["success"]
        self.message = data["message"]
        self.data = data["data"]


class TypeChatLanguageModel():
    def __init__(self, model_name, api_base=None, org_key=None, api_key=None, retryMaxAttempts=3, retryPauseSec=5, use_chat = False, use_json_mode=False):
        # Why 30? Because we try three times and OpenAI resets every 60 seconds...
        self._retryMaxAttempts = retryMaxAttempts
        self._retryPauseSec = retryPauseSec
        self._model_name = model_name
        self._use_chat = use_chat
        self._use_json_mode = use_json_mode

        self._client = None
        if api_base:
            self._client = openai.OpenAI(
                organization=org_key if org_key else None,
                api_key=api_key if api_key else "EMPTY",
                base_url=api_base
            )
        else:
            self._client = openai.OpenAI(
                organization=org_key if org_key else None,
                api_key=api_key if api_key else "EMPTY"
            )
    
    def useChat(self):
        return self._use_chat

    def complete(self, prompt, return_query=False):
        # print("#"*150)
        # print("#"*150)
        # print(prompt)
        # print("%"*150)
        # print("%"*150)
        result = TypeChatResult()
        for i in range(self._retryMaxAttempts):
            try:
                if self._use_chat:
                    msgs = []
                    for msg in prompt:
                        if msg[0] == "s":
                            msgs.append({"role": "system", "content": msg[1]})
                        elif msg[0] == "u":
                            msgs.append({"role": "user", "content": msg[1]})
                        elif msg[0] == "a":
                            msgs.append({"role": "assistant", "content": msg[1]})
                    if return_query:
                        return msgs
                    completion = None
                    if self._use_json_mode:
                        completion = self._client.chat.completions.create(
                            model=self._model_name,
                            messages = msgs,
                            max_tokens=512,
                            temperature=0.0,
                            response_format={ "type": "json_object" },
                        )
                    else:
                        completion = self._client.chat.completions.create(
                            model=self._model_name,
                            messages = msgs,
                            max_tokens=512,
                            temperature=0.0,
                        )
                    result.success = True
                    result.data = completion.choices[0].message.content
                    result.message = ""
                    return result
                else:
                    completion = self._chat.completions.create(
                        model=self._model_name,
                        prompt=prompt,
                        max_tokens=256,
                        temperature=0.0
                    )
                    result.success = True
                    result.data = completion["choices"][0]["text"]
                    result.message = ""
                    # print("LLM:", result.data)
                    return result
            except Exception as error:
                print("Error:", error, "Retry count:", self._retryMaxAttempts-i)
                result.success = False
                result.message = str(error)
                time.sleep(self._retryPauseSec)
        return result

class TypeChatJsonTranslator():
    def __init__(self, model, validator, attemptRepair=True, stripNulls=False):
        self._model = model
        self._validator = validator
        self._attemptRepair = attemptRepair
        self._stripNulls = stripNulls

    def createRequestPrompt(self, request, image=None, free_form=False):
        prompt = None
        if self._model.useChat():
            prompt = []
            system = f"Respond to the user request with JSON objects of type {self._validator.getTypeName()} according to the following TypeScript definitions:\n" 
            system += f"```\n{self._validator.getSchema()}```\n"
            prompt.append(("s", system))
            if type(request) == list:
                for mid, msg in enumerate(request):
                    if mid == 0 and msg["role"] == "user":
                        pr = f"The following is my request:\n"
                        pr += f"\"\"\"\n{msg['content']}\n\"\"\"\n"
                        prompt.append(("u", pr))
                        if image:
                            pr = [{"image": base64.b64encode(image.read()).decode('ascii')}, pr]
                    elif msg["role"] == "user":
                        prompt.append(("u", msg["content"]))
                    elif msg["role"] == "assistant":
                        prompt.append(("a", msg["content"]))
                uhandle, msg = prompt[-1]
                prompt[-1] = (uhandle, msg + " Answer my request translated into a JSON object with 2 spaces of indentation and no properties with the value undefined")
            else:
                user = f"The following is my request:\n"
                user += f"\"\"\"\n{request}\n\"\"\"\n"
                user += "Answer my request translated into a JSON object with 2 spaces of indentation and no properties with the value undefined"
                if image:
                    user = [{"image": base64.b64encode(image.read()).decode('ascii')}, user]
                prompt.append(("u", user))
        else:
            prompt = f"You are a service that translates user requests into JSON objects of type {self._validator.getTypeName()} according to the following TypeScript definitions:\n" 
            prompt += f"```\n{self._validator.getSchema()}```\n"
            prompt += f"The following is a user request:\n"
            prompt += f"\"\"\"\n{request}\n\"\"\"\n"
            prompt += f"The following is the user request translated into a JSON object with 2 spaces of indentation and no properties with the value undefined:\n"
        return prompt

    def createRepairPrompt(self, validationError):
        prompt = None
        if self._model.useChat():
            prompt = "The JSON object is invalid for the following reason:\n"
            prompt += f"\"\"\"\n{validationError}\n\"\"\"\n" 
            prompt += "Please correct the error and provide the revised JSON object.\n" 
        else:
            prompt = "The JSON object is invalid for the following reason:\n"
            prompt += f"\"\"\"\n{validationError}\n\"\"\"\n" 
            prompt += "The following is a revised JSON object:\n"
        return prompt
    
    def _getFirstValidJSON(self, responseText):
        res = TypeChatResult()
        start_index = responseText.find('{')
        pref_index = start_index + 1
        parentheses = 1
        for _ in range(50):
            next_open = responseText.find('{', pref_index)
            next_close = responseText.find('}', pref_index)
            if next_open > 0 and next_open < next_close:
                parentheses += 1
                pref_index = next_open + 1
            elif next_close > 0 and next_open > next_close:
                parentheses -= 1
                pref_index = next_close + 1
            elif next_open < 0 and next_close > 0:
                parentheses -= 1
                pref_index = next_close + 1
            elif next_open < 0 and next_close < 0 and parentheses != 0:
                res.success = False
                res.message = "Invalid JSON"
                return res
            if parentheses == 0:
                res.success = True
                res.data = responseText[start_index:pref_index]
                return res
            if parentheses > 25:
                res.success = False
                res.message = "Max JSON parentheses depth of 25 reached"
                return res
        res.success = False
        res.message = "Max JSON search depth of 50 reached"
        return res
            

    def translate(self, request, image=None, return_query=False, free_form = False):
        result = TypeChatResult()
        if type(request) == list:
            # Check if the first and last message is a user message, and if user and assistant messages alternate
            if request[0]["role"] != "user" or request[-1]["role"] != "user":
                result.success = False
                result.message = "The first and last message have to be user messages"
                return result
            for i in range(1, len(request)-1, 2):
                if request[i]["role"] != "assistant":
                    result.success = False
                    result.message = "User and assistant messages have to alternate"
                    return 
            if len(request) < 1:
                result.success = False
                result.message = "Request has to be a list of messages with at least one user message"
                return result
        
        prompt = self.createRequestPrompt(request, image, free_form=free_form)
        attemptRepair = self._attemptRepair        
        while True:
            response = self._model.complete(prompt, return_query=return_query)
            if return_query:
                return response
            if not response.success:
                return response
            responseText = response.data

            jsonText = self._getFirstValidJSON(responseText)
            if not jsonText.success:
                return jsonText
            jsonText = jsonText.data
            # print("-"*150)
            # print("Clean JSON", jsonText)
            # print("-"*150)
            # startIndex = responseText.index('{')
            # endIndex = len(responseText) - 1 - responseText[::-1].index('}')
            # if not (startIndex >= 0 and endIndex > startIndex):
            #     result.success = False
            #     result.message = f"Response is not JSON:\n{responseText}"
            #     return result
            # jsonText = responseText[startIndex:endIndex+1]


            validation = self._validator.validate(jsonText)
            # print("Validation:", jsonText, validation.success, validation.message)

            if validation.success:
                return validation
            
            if not attemptRepair:
                result.success = False
                result.message = f"JSON validation failed:\n{validation.message}\n{jsonText}"
                return result
            
            if self._model.useChat():
                prompt.append(("a",f"```\n{jsonText}\n```"))
                prompt.append(("u",self.createRepairPrompt(validation.message)))
            else:
                prompt += f"```\n{jsonText}\n```\n{self.createRepairPrompt(validation.message)}"
            attemptRepair = False
    
    def validateExample(self, jsonText):
        validation = self._validator.validate(jsonText)
        return validation

class TypeChatJsonValidator():
    def __init__(self, schema, name, basedir):
        self._schema = schema
        self._typeName = name
        self._stripNulls = False
        self._basedir = basedir

        self._options = [
            "--target", "es2021",
            "--lib", "es2021",
            "--module", "node16",
            # "--types", "['node']",
            "--esModuleInterop", "true",
            "--outDir", "./schemas/out",
            "--skipLibCheck", "true",
            "--strict", "true",
            "--exactOptionalPropertyTypes", "true",
            "--declaration", "true",
            # "--noLib", "true",
            "--noEmit",
        ]

        self._rootProgram = self._createProgramFromModuleText("")
    
    def getTypeName(self):
        return self._typeName
    
    def getSchema(self):
        return self._schema
    
    def _createProgramFromModuleText(self, moduleText, oldProgram=None):
        with open(f"{self._basedir}/out/lib.d.ts", "w") as fh:
            fh.write("interface Array<T> { length: number, [n: number]: T }\ninterface Object { toString(): string }\ninterface Function { prototype: unknown }\ninterface CallableFunction extends Function {}\ninterface NewableFunction extends Function {}\ninterface String { readonly length: number }\ninterface Boolean { valueOf(): boolean }\ninterface Number { valueOf(): number }\ninterface RegExp { test(string: string): boolean }")
        
        with open(f"{self._basedir}/out/schema.ts", "w") as fh:
            fh.write(self._schema)

        with open(f"{self._basedir}/out/json.ts", "w") as fh:
            fh.write(moduleText)        
        
    def createModuleTextFromJson(self, jsonObject):
        code = f"import {{ {self._typeName} }} from './schema';\nconst json: {self._typeName} = {json.dumps(jsonObject)};"
        result = TypeChatResult()
        result.success = True
        result.data = code
        return result

    def validate(self, jsonText):
        jsonObject = None
        try:
            jsonObject = json.loads(jsonText)
        except Exception as error:
            result = TypeChatResult()
            result.success = False
            result.message = error
            return result

        if self._stripNulls:
            jsonObject = self._stripNone(jsonObject)

        moduleResult = self.createModuleTextFromJson(jsonObject)  

        if not moduleResult.success:
            return moduleResult

        self._createProgramFromModuleText(moduleResult.data, self._rootProgram)
        result = self.getSyntacticDiagnostics()

        if not result.success:
            return result
        
        result.data = jsonObject
        return result
    
    def getSyntacticDiagnostics(self):
        command = ["tsc", f"{self._basedir}/out/json.ts", f"{self._basedir}/out/lib.d.ts", f"{self._basedir}/out/schema.ts"] + self._options
        res = subprocess.run(command, stdout=subprocess.PIPE, text=True)
        # print(res)
        result = TypeChatResult()
        result.success = res.returncode == 0
        if not result.success:
            result.message = res.stdout
        return result

    def _stripNone(self, value):
        """
        Recursively remove all None values from dictionaries and lists, and returns
        the result as a new dictionary or list.
        """
        if isinstance(value, list):
            return [self._stripNone(x) for x in value if x is not None]
        elif isinstance(value, dict):
            return {
                key: self._stripNone(val)
                for key, val in value.items()
                if val is not None
            }
        else:
            return value

class TypeChat():
    def createLanguageModel(self, model, base_url=None, org_key=None, api_key=None, use_json_mode=False):
        self._model = TypeChatLanguageModel(model, base_url, org_key, api_key, use_chat=True, use_json_mode=use_json_mode)
    
    def createJsonValidator(self, schema, name, basedir):
        validator = TypeChatJsonValidator(schema, name, basedir=basedir)
        return validator
    
    def loadSchema(self, path=None, schema=None):
        if path is None and schema is None:
            print("Provide a path to a schema or a schema directly")
            return False
        
        if path:
            if not os.path.exists(path):
                print(f"Scheme '{path}' not found!")
                exit(0)
                return False
        
            with open(path, "r", encoding="utf-8") as fh:
                self._schema = fh.read()
        elif schema:
            self._schema = schema
        return True
    
    def createJsonTranslator(self, name=None, basedir="./code/typechat/schemas"):
        self._validator = self.createJsonValidator(self._schema, name, basedir)
        translator = TypeChatJsonTranslator(self._model, self._validator, attemptRepair=True, stripNulls=False)
        return translator