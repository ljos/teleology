import pyodideModule from "pyodide/pyodide.js";
import { TextLineStream } from "@std/streams/text-line-stream";

const pyodide = await pyodideModule.loadPyodide();

pyodide.mountNodeFS("/home/pyodide/data", "data");


pyodide.setStdin({ error: true });

let capturedStdout = "";
function captureStdout(output) {
    capturedStdout += output;
}

pyodide.setStdout({ batched: captureStdout });


let capturedStderr = "";
function captureStderr(output) {
    capturedStderr += output;
}

await pyodide.loadPackage([
    "matplotlib",
    "networkx",
    "numpy",
    "pandas",
    "scipy",
    "uncertainties",
], { messageCallback: () => {} });

const stdin = Deno.stdin.readable
    .pipeThrough(new TextDecoderStream())
    .pipeThrough(new TextLineStream());

for await (const line of stdin) {
    capturedStdout = "";
    capturedStderr = "";

    let output;
    let input;

    try {

        input = JSON.parse(line);

    } catch (error) {
        output = JSON.stringify({error: "Invalid JSON: " + error.message});
        console.log(output);
        continue;
    }

    if (typeof input !== 'object' || input === null) {
        output = JSON.stringify({error: "INput is not a JSON object"});
        console.log(output);
    }

    if (input.exit) {
        Deno.exit(0);
    }

    try {
        const result = await pyodide.runPythonAsync(input.code || "");
        output = JSON.stringify({
            output: String(result || ""),
            stdout: capturedStdout,
            stderr: capturedStderr,
        });
    } catch (error) {
        output = JSON.stringify({
            error: error.message.trim().split("\n").pop() || "",
            stdout: capturedStdout,
            stderr: capturedStderr,
        });
    }

    console.log(output);
}
