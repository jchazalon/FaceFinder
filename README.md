# FaceFinder
FaceFinder is a Python tool that helps you quickly locate all photos featuring a specific friend in your image collection.

## Installation
To install the required dependencies, use the following command:
```
uv sync
```

## Usage

FaceFinder provides two main commands: `index` and `query`.

### Indexing Faces
To index all faces in a folder:

```
uv run facefinder index <input_folder> <index_folder>
```

- `<input_folder>`: Path to the folder containing images.
- `<index_folder>`: Path to the folder where the index and metadata will be stored.


### Querying Faces
To query a sample image:

```
uv run facefinder query <image> <index_folder> <output_folder> [--topk <number>]
```

- `<image>`: Path to the sample image.
- `<index_folder>`: Path to the folder containing the index and metadata.
- `<output_folder>`: Path to the folder where query results will be stored.
- `--topk`: (Optional) Number of matches to return (default: 50).

💡 You can use one of the generated thumbnails as query.

## Limitations and possible improvements
- It has the whole Internet as dependency…
- The indexing is slow as hell.
- The result template is hugly.
- It is not possible to index files incrementally: you cannot interrupt the indexing and continue later, you cannot add files to an already- computed index.
- The result list is not filtered by confidence, so first results should be OK, but the rest may not.
- Retreival quality may be improved.
