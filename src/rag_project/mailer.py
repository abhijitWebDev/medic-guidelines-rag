"""Outbound email, over Amazon SES's SMTP interface.

SMTP rather than the SES API, and `smtplib` rather than boto3, for the reason
that runs through this project: an SDK is a dependency you keep patched
forever, and botocore ships the service catalogue for every AWS API -- tens of
megabytes in a bundle that has a size limit -- to make one SendEmail call. The
stdlib does this in forty lines. It is the same trade already made for scrypt
over argon2 and for the hand-written LanceDB client.

**Configuring a host is what turns verification on.** With `SES_SMTP_HOST` and
`MAIL_FROM` unset there is no mailer, and `Settings.verification_required` is
false, so accounts work the moment they are created. A fresh clone and the test
suite need that; more importantly, an instance that *cannot* send must never
demand that people click a link it will never deliver. That is not a stricter
deployment, it is one nobody can sign into.

Sending happens inside the request rather than in a background task. On a
serverless platform the instance can be frozen the moment a response is
returned, so "send it after we reply" is a good way to lose the email that the
account depends on. Signup pays the few hundred milliseconds.

**SES starts every account in the sandbox**, where it will only deliver to
addresses you have verified in the console. Until you request production
access, signing up with an unverified address gets a clean "we could not send
it" rather than a mystery -- see the README.
"""

from __future__ import annotations

import smtplib
import ssl
import sys
from email.message import EmailMessage
from email.utils import formataddr

from .config import get_settings


class MailError(RuntimeError):
    """The message could not be handed to SES."""


def enabled() -> bool:
    return get_settings().mail_enabled


def send(to: str, subject: str, body: str) -> None:
    """Deliver one plain-text message, or raise MailError.

    Plain text only. An HTML mail is a second body to keep in sync, it is what
    spam filters weigh hardest, and every message this app sends is one
    sentence and one link.
    """
    s = get_settings()
    if not s.mail_enabled:
        raise MailError("email is not configured (SES_SMTP_HOST / MAIL_FROM)")

    msg = EmailMessage()
    msg["From"] = formataddr((s.mail_from_name, s.mail_from))
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    context = ssl.create_default_context()
    try:
        # Port decides the handshake: 465 is implicit TLS, everything else is
        # STARTTLS. Inferring it beats a separate flag that can disagree with
        # the port and produce a hang instead of an error.
        if s.ses_smtp_port == 465:
            with smtplib.SMTP_SSL(
                s.ses_smtp_host, s.ses_smtp_port, timeout=s.mail_timeout_s, context=context
            ) as smtp:
                _login_and_send(smtp, msg)
        else:
            with smtplib.SMTP(
                s.ses_smtp_host, s.ses_smtp_port, timeout=s.mail_timeout_s
            ) as smtp:
                smtp.starttls(context=context)
                _login_and_send(smtp, msg)
    except (smtplib.SMTPException, OSError, ssl.SSLError) as e:
        # Logged here because callers turn this into a user-facing sentence and
        # would otherwise swallow the only description of what SES objected to.
        print(f"mail: could not send to {to}: {e}", file=sys.stderr)
        raise MailError(str(e)) from e


def _login_and_send(smtp: smtplib.SMTP, msg: EmailMessage) -> None:
    s = get_settings()
    if s.ses_smtp_user:
        smtp.login(s.ses_smtp_user, s.ses_smtp_password)
    smtp.send_message(msg)


# --- the two messages this application sends -----------------------------
#
# Written out rather than templated. There are two of them, they are four lines
# each, and a template engine to fill two holes is a dependency to keep patched
# for the rest of the project's life.


def send_verification(to: str, link: str) -> None:
    send(
        to,
        "Confirm your email",
        "Confirm this address to start asking questions:\n\n"
        f"{link}\n\n"
        "The link is good for three days. If you did not create an account, "
        "ignore this message -- nothing was activated and the address will not "
        "be used again.\n",
    )


def send_password_reset(to: str, link: str) -> None:
    send(
        to,
        "Reset your password",
        "Use this link to choose a new password:\n\n"
        f"{link}\n\n"
        "The link is good for one hour and can be used once. Setting a new "
        "password signs the account out everywhere.\n\n"
        "If you did not ask for this, ignore this message -- your current "
        "password still works and nothing has changed.\n",
    )
