"""Sync commands to specific guild (instant)."""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import discord
from dotenv import load_dotenv
from discord_bot.commands import VoiceBotCommands

load_dotenv()

GUILD_ID = int(os.getenv("DISCORD_GUILD_ID", "646779509529509900"))

async def main():
    intents = discord.Intents.default()
    client = discord.Client(intents=intents)
    tree = discord.app_commands.CommandTree(client)

    @client.event
    async def on_ready():
        print(f"Connected as {client.user}")

        # Get guild
        guild = discord.Object(id=GUILD_ID)
        print(f"Syncing to guild ID: {GUILD_ID}")

        # Add command group
        commands = VoiceBotCommands(client)
        tree.add_command(commands)

        # Sync to specific guild (instant)
        synced = await tree.sync(guild=guild)

        print(f"\n✓ SUCCESS! Synced {len(synced)} command(s) to your guild:")
        for cmd in synced:
            print(f"  /{cmd.name}")

        print(f"\nCommands should appear instantly in Discord!")
        print(f"Try typing /jarvis in your server now.")

        await client.close()

    try:
        await client.start(os.getenv("DISCORD_TOKEN"))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    asyncio.run(main())
