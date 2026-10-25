"""
Maintenance commands.

    python -m backend.cli reset-owner   Forget the owner passphrase (data and API keys are kept).
                                        Restart the server to get a new setup code.
    python -m backend.cli migrate       Apply database migrations.
"""

from __future__ import annotations

import asyncio
import sys

from backend.db import database
from backend.db.models import Owner


async def reset_owner() -> None:
    await database.upgrade()
    async with database.session() as s:
        owner = await s.get(Owner, 1)
        if owner is None:
            print("No owner passphrase is configured; nothing to reset.")
            return
        await s.delete(owner)
    print("Owner passphrase removed. Restart the server and use the printed setup code to set a new one.")


async def migrate() -> None:
    await database.upgrade()
    print("Database is up to date.")


COMMANDS = {"reset-owner": reset_owner, "migrate": migrate}


def main(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0] not in COMMANDS:
        print(__doc__)
        return 2
    asyncio.run(COMMANDS[argv[0]]())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
