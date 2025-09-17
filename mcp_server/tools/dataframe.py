from common.connection import RedisConnectionManager
from redis.exceptions import RedisError
from common.server import mcp
import base64
import pandas as pd
import pickle
from pandas import DataFrame
import asyncio
import redis

# @mcp.tool()
# async def getdf(key: str) -> DataFrame:
#     """Get a Redis string value.
#
#     Args:
#         key (str): The key to retrieve.
#
#     Returns:
#         str: The stored value or an error message.
#     """
#     try:
#        # r = RedisConnectionManager.get_connection()
#         r = redis.StrictRedis(
#             host='amaaaaaawe6j4fqaxqkbzpawdnhcyt2brjexcaamvemvgpbmhotsozgj46qa-p.redis.us-chicago-1.oci.oraclecloud.com',
#             ssl=True, decode_responses=False, port=6379)
#         key = f"idata:{key}:latest"
#         print(f'get keys {key}')
#         raw = r.get(key)
#         print(raw)
#         #df = pd.DataFrame()
#         if raw is not None:
#             df = pickle.loads(raw)
#         else:
#             df = None
#         return df if df else f"Key {key} does not exist"
#     except RedisError as e:
#         return f"Error retrieving key {key}: {str(e)}"
#


@mcp.tool()
async def getdf(key: str)  -> dict:
    """Get a Redis string value.

    Args:
        key (str): The key to retrieve.

    Returns:
        str: The stored value or an error message.
    """
    r = RedisConnectionManager.get_connection()

    print(r)
    # r = redis.StrictRedis(
    # host='amaaaaaawe6j4fqaxqkbzpawdnhcyt2brjexcaamvemvgpbmhotsozgj46qa-p.redis.us-chicago-1.oci.oraclecloud.com',
    # ssl=True, decode_responses=False, port=6379)

    key_name = f"idata:{key}:idata:latest"
    raw = r.get(key_name)

    if raw is None:
        return f"Key {key_name} does not exist"

    # ensure we have bytes, even if client auto-decoded
    if isinstance(raw, str):
        raw = raw.encode("latin-1")

    try:
        df = pickle.loads(raw)
    except Exception as e:
        return f"Error unpickling data from {key_name}: {e}"

    # now handle an empty DataFrame explicitly
    if isinstance(df, DataFrame) and df.empty:
        return DataFrame({"message": [f"Key {key_name} exists but contains an empty DataFrame"]})

    # finally, serialize the DataFrame for return
    # you can choose CSV, JSON, etc. Here’s JSON split-orient:
    return df

async def main():
    df = await getdf("02e4b9e5-5e92-4836-b589-7536266c7baa")
    print(df)

if __name__ == "__main__":
    asyncio.run(main())