"""The API contrib app: the service layer over the framework (documents 02 and 13).

Add ``"mactf.contrib.api"`` to ``INSTALLED_APPS`` to get ``api/v1`` routes, the
``serve`` command and the ``mactf.api.*`` system checks. Remove it and the
framework is untouched: ``run``, ``check`` and ``migrate`` keep working.
"""

default_app_config = "mactf.contrib.api.apps.ApiConfig"
