import uvicorn
import argparse
from eto_sm_module import eto_sm_module

def main():
    parser = argparse.ArgumentParser(description="Start FastAPI app with GGE account")
    parser.add_argument("--account_gge", required=True, help="GGE account")
    parser.add_argument("--auth_file", required=True, help="Name GGE Json file")
    args=parser.parse_args()
    app=eto_sm_module.create_app(args.account_gge, args.auth_file)
    uvicorn.run(app, host="127.0.0.1",port=8000)

if __name__ == "__main__":
    main()