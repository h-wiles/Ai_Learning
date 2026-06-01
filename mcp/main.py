from mcp.server.fastmcp import FastMCP
import tools

mcp = FastMCP("host info mcp")
mcp.add_tool(tools.get_host_info)    # 注册工具函数

# 另一种使用装饰器定义的工具，
@mcp.tool()
def foo():
    return ""

def main():
    mcp.run("stdio") # stdio是标准的输入输出，另一种方式sse，可以部署云端可远程调用


if __name__ == "__main__":
    main()