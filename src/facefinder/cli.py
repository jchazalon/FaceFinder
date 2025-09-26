from facefinder.index import build_index, query

# ----------------------------
# CLI
# ----------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Face indexing and querying tool with HTML preview. First index faces in a folder, then query with a sample image.")
    subparsers = parser.add_subparsers(dest="command")

    # Index command
    p_index = subparsers.add_parser("index", help="Index all faces in a folder")
    p_index.add_argument("input_folder", help="Path to folder with images")
    p_index.add_argument("index_folder", help="Path to folder to store index and metadata")

    # Query command
    p_query = subparsers.add_parser("query", help="Query a sample image")
    p_query.add_argument("image", help="Path to sample image")
    p_query.add_argument("index_folder", help="Path to folder which contains index and metadata")
    p_query.add_argument("output_folder", help="Path to folder to store query results")
    p_query.add_argument("--topk", type=int, default=50, help="Number of matches to return")

    args = parser.parse_args()

    if args.command == "index":
        build_index(args.input_folder, args.index_folder)
    elif args.command == "query":
        query(args.image, args.index_folder, args.output_folder, args.topk)
    else:
        parser.print_help()
