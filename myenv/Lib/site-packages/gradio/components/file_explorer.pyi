"""gr.FileExplorer() component"""

from __future__ import annotations

import itertools
import os
import re
from pathlib import Path
from typing import Any, Callable, List, Literal

from gradio_client.documentation import document, set_documentation_group

from gradio.components.base import Component, server
from gradio.data_classes import GradioRootModel

set_documentation_group("component")


class FileExplorerData(GradioRootModel):
    root: List[List[str]]

from gradio.events import Dependency

@document()
class FileExplorer(Component):
    """
    Creates a file explorer component that allows users to browse and select files on the machine hosting the Gradio app.
    Preprocessing: passes the selected file or directory as a {str} path (relative to root) or {list[str}} depending on `file_count`
    Postprocessing: expects function to return a {str} path to a file, or {List[str]} consisting of paths to files.
    Examples-format: a {str} path to a local file that populates the component.
    Demos: zip_to_json, zip_files
    """

    EVENTS = ["change"]
    data_model = FileExplorerData

    def __init__(
        self,
        glob: str = "**/*.*",
        *,
        value: str | list[str] | Callable | None = None,
        file_count: Literal["single", "multiple"] = "multiple",
        root: str | Path = ".",
        ignore_glob: str | None = None,
        label: str | None = None,
        every: float | None = None,
        show_label: bool | None = None,
        container: bool = True,
        scale: int | None = None,
        min_width: int = 160,
        height: int | float | None = None,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        render: bool = True,
    ):
        """
        Parameters:
            glob: The glob-style pattern used to select which files to display, e.g. "*" to match all files, "*.png" to match all .png files, "**/*.txt" to match any .txt file in any subdirectory, etc. The default value matches all files and folders recursively. See the Python glob documentation at https://docs.python.org/3/library/glob.html for more information.
            value: The file (or list of files, depending on the `file_count` parameter) to show as "selected" when the component is first loaded. If a callable is provided, it will be called when the app loads to set the initial value of the component. If not provided, no files are shown as selected.
            file_count: Whether to allow single or multiple files to be selected. If "single", the component will return a single absolute file path as a string. If "multiple", the component will return a list of absolute file paths as a list of strings.
            root: Path to root directory to select files from. If not provided, defaults to current working directory.
            ignore_glob: The glob-tyle pattern that will be used to exclude files from the list. For example, "*.py" will exclude all .py files from the list. See the Python glob documentation at https://docs.python.org/3/library/glob.html for more information.
            label: The label for this component. Appears above the component and is also used as the header if there are a table of examples for this component. If None and used in a `gr.Interface`, the label will be the name of the parameter this component is assigned to.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            container: If True, will place the component in a container - providing some extra padding around the border.
            scale: relative width compared to adjacent Components in a Row. For example, if Component A has scale=2, and Component B has scale=1, A will be twice as wide as B. Should be an integer.
            min_width: minimum pixel width, will wrap if not sufficient screen space to satisfy this value. If a certain scale value results in this Component being narrower than min_width, the min_width parameter will be respected first.
            height: The maximum height of the file component, specified in pixels if a number is passed, or in CSS units if a string is passed. If more files are uploaded than can fit in the height, a scrollbar will appear.
            interactive: if True, will allow users to upload a file; if False, can only be used to display files. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
            render: If False, component will not render be rendered in the Blocks context. Should be used if the intention is to assign event listeners now but render the component later.
        """
        self.root = os.path.abspath(root)
        self.glob = glob
        self.ignore_glob = ignore_glob
        valid_file_count = ["single", "multiple", "directory"]
        if file_count not in valid_file_count:
            raise ValueError(
                f"Invalid value for parameter `type`: {type}. Please choose from one of: {valid_file_count}"
            )
        self.file_count = file_count
        self.height = height

        super().__init__(
            label=label,
            every=every,
            show_label=show_label,
            container=container,
            scale=scale,
            min_width=min_width,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            render=render,
            value=value,
        )

    def example_inputs(self) -> Any:
        return ["Users", "gradio", "app.py"]

    def preprocess(self, payload: FileExplorerData | None) -> list[str] | str | None:
        if payload is None:
            return None

        if self.file_count == "single":
            if len(payload.root) > 1:
                raise ValueError(
                    f"Expected only one file, but {len(payload.root)} were selected."
                )
            elif len(payload.root) == 0:
                return None
            else:
                return self._safe_join(payload.root[0])
        files = []
        for file in payload.root:
            file_ = self._safe_join(file)
            files.append(file_)
        return files

    def _strip_root(self, path):
        if path.startswith(self.root):
            return path[len(self.root) + 1 :]
        return path

    def postprocess(self, value: str | list[str] | None) -> FileExplorerData | None:
        if value is None:
            return None

        files = [value] if isinstance(value, str) else value
        root = []
        for file in files:
            root.append(self._strip_root(file).split(os.path.sep))

        return FileExplorerData(root=root)

    @server
    def ls(self, value=None) -> list[dict[str, str]] | None:
        """
        Parameters:
            value: file path as a list of strings for each directory level relative to the root.
        Returns:
            tuple of list of files in directory, then list of folders in directory
        """

        def expand_braces(text, seen=None):
            if seen is None:
                seen = set()

            spans = [m.span() for m in re.finditer("{[^{}]*}", text)][::-1]
            alts = [text[start + 1 : stop - 1].split(",") for start, stop in spans]

            if len(spans) == 0:
                if text not in seen:
                    yield text
                seen.add(text)

            else:
                for combo in itertools.product(*alts):
                    replaced = list(text)
                    for (start, stop), replacement in zip(spans, combo):
                        replaced[start:stop] = replacement

                    yield from expand_braces("".join(replaced), seen)

        def make_tree(files):
            tree = []
            for file in files:
                parts = file.split(os.path.sep)
                make_node(parts, tree)
            return tree

        def make_node(parts, tree):
            _tree = tree
            for i in range(len(parts)):
                if _tree is None:
                    continue
                if i == len(parts) - 1:
                    type = "file"
                    _tree.append({"path": parts[i], "type": type, "children": None})
                    continue
                type = "folder"
                j = next(
                    (index for (index, v) in enumerate(_tree) if v["path"] == parts[i]),
                    None,
                )
                if j is not None:
                    _tree = _tree[j]["children"]
                else:
                    _tree.append({"path": parts[i], "type": type, "children": []})
                    _tree = _tree[-1]["children"]

        files: list[Path] = []
        for result in expand_braces(self.glob):
            files += list(Path(self.root).resolve().glob(result))

        files = [f for f in files if f != Path(self.root).resolve()]

        ignore_files = []
        if self.ignore_glob:
            for result in expand_braces(self.ignore_glob):
                ignore_files += list(Path(self.root).resolve().glob(result))
            files = list(set(files) - set(ignore_files))

        files_with_sep = []
        for f in files:
            file = str(f.relative_to(self.root))
            if f.is_dir():
                file += os.path.sep
            files_with_sep.append(file)

        tree = make_tree(files_with_sep)
        return tree

    def _safe_join(self, folders):
        combined_path = os.path.join(self.root, *folders)
        absolute_path = os.path.abspath(combined_path)
        if os.path.commonprefix([self.root, absolute_path]) != os.path.abspath(
            self.root
        ):
            raise ValueError("Attempted to navigate outside of root directory")
        return absolute_path

    
    def change(self,
        fn: Callable | None,
        inputs: Component | Sequence[Component] | set[Component] | None = None,
        outputs: Component | Sequence[Component] | None = None,
        api_name: str | None | Literal[False] = None,
        scroll_to_output: bool = False,
        show_progress: Literal["full", "minimal", "hidden"] = "full",
        queue: bool | None = None,
        batch: bool = False,
        max_batch_size: int = 4,
        preprocess: bool = True,
        postprocess: bool = True,
        cancels: dict[str, Any] | list[dict[str, Any]] | None = None,
        every: float | None = None,
        trigger_mode: Literal["once", "multiple", "always_last"] | None = None,
        js: str | None = None,
        concurrency_limit: int | None | Literal["default"] = "default",
        concurrency_id: str | None = None,
        show_api: bool = True) -> Dependency:
        """
        Parameters:
            fn: the function to call when this event is triggered. Often a machine learning model's prediction function. Each parameter of the function corresponds to one input component, and the function should return a single value or a tuple of values, with each element in the tuple corresponding to one output component.
            inputs: List of gradio.components to use as inputs. If the function takes no inputs, this should be an empty list.
            outputs: List of gradio.components to use as outputs. If the function returns no outputs, this should be an empty list.
            api_name: Defines how the endpoint appears in the API docs. Can be a string, None, or False. If False, the endpoint will not be exposed in the api docs. If set to None, the endpoint will be exposed in the api docs as an unnamed endpoint, although this behavior will be changed in Gradio 4.0. If set to a string, the endpoint will be exposed in the api docs with the given name.
            scroll_to_output: If True, will scroll to output component on completion
            show_progress: If True, will show progress animation while pending
            queue: If True, will place the request on the queue, if the queue has been enabled. If False, will not put this event on the queue, even if the queue has been enabled. If None, will use the queue setting of the gradio app.
            batch: If True, then the function should process a batch of inputs, meaning that it should accept a list of input values for each parameter. The lists should be of equal length (and be up to length `max_batch_size`). The function is then *required* to return a tuple of lists (even if there is only 1 output component), with each list in the tuple corresponding to one output component.
            max_batch_size: Maximum number of inputs to batch together if this is called from the queue (only relevant if batch=True)
            preprocess: If False, will not run preprocessing of component data before running 'fn' (e.g. leaving it as a base64 string if this method is called with the `Image` component).
            postprocess: If False, will not run postprocessing of component data before returning 'fn' output to the browser.
            cancels: A list of other events to cancel when this listener is triggered. For example, setting cancels=[click_event] will cancel the click_event, where click_event is the return value of another components .click method. Functions that have not yet run (or generators that are iterating) will be cancelled, but functions that are currently running will be allowed to finish.
            every: Run this event 'every' number of seconds while the client connection is open. Interpreted in seconds. Queue must be enabled.
            trigger_mode: If "once" (default for all events except `.change()`) would not allow any submissions while an event is pending. If set to "multiple", unlimited submissions are allowed while pending, and "always_last" (default for `.change()` event) would allow a second submission after the pending event is complete.
            js: Optional frontend js method to run before running 'fn'. Input arguments for js method are values of 'inputs' and 'outputs', return should be a list of values for output components.
            concurrency_limit: If set, this is the maximum number of this event that can be running simultaneously. Can be set to None to mean no concurrency_limit (any number of this event can be running simultaneously). Set to "default" to use the default concurrency limit (defined by the `default_concurrency_limit` parameter in `Blocks.queue()`, which itself is 1 by default).
            concurrency_id: If set, this is the id of the concurrency group. Events with the same concurrency_id will be limited by the lowest set concurrency_limit.
            show_api: whether to show this event in the "view API" page of the Gradio app, or in the ".view_api()" method of the Gradio clients. Unlike setting api_name to False, setting show_api to False will still allow downstream apps to use this event. If fn is None, show_api will automatically be set to False.
        """
        ...