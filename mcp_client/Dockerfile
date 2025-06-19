FROM langchain/langgraph-api:3.11



# -- Installing local requirements --
ADD requirements.txt /deps/__outer_mcp_client/mcp_client/requirements.txt
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -r /deps/__outer_mcp_client/mcp_client/requirements.txt
# -- End of local requirements install --

# -- Adding non-package dependency mcp_client --
ADD . /deps/__outer_mcp_client/mcp_client
RUN set -ex && \
    for line in '[project]' \
                'name = "mcp_client"' \
                'version = "0.1"' \
                '[tool.setuptools.package-data]' \
                '"*" = ["**/*"]'; do \
        echo "$line" >> /deps/__outer_mcp_client/pyproject.toml; \
    done
# -- End of non-package dependency mcp_client --

ENV LANGSERVE_GRAPHS='{"askdata_getinsights": "/deps/__outer_mcp_client/mcp_client/app/getinsights/askdata_getinsights.py:askdata_getinsights", "agent_supervisor": "/deps/__outer_mcp_client/mcp_client/app/crossdomains/agent_supervisor.py:agent_supervisor", "agent_supervisor_swarm": "/deps/__outer_mcp_client/mcp_client/app/crossdomains/agent_supervisor_swarm.py:run_swarm", "agent_bob_the_pirate": "/deps/__outer_mcp_client/mcp_client/app/crossdomains/agent_swarm.py:bob_the_pirate"}'

# -- Removing pip from the final image ~<:===~~~ --
RUN pip uninstall -y pip setuptools wheel &&     rm -rf /usr/local/lib/python*/site-packages/pip* /usr/local/lib/python*/site-packages/setuptools* /usr/local/lib/python*/site-packages/wheel* &&     find /usr/local/bin -name "pip*" -delete
# -- End of pip removal --

WORKDIR /deps/__outer_mcp_client/mcp_client