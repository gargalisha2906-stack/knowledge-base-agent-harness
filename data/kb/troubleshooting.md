# Troubleshooting Guide

Login problems: first confirm the email address is the one invited to the
workspace; invitations expire after 7 days and an admin must resend. Password
reset links expire after 1 hour. If 2FA codes fail, check that the device clock
is synchronised; a drift of more than 30 seconds breaks TOTP codes.

Slow dashboards: ticket counts above 50,000 can make the default views slow.
Create a filtered view by status or team, which uses precomputed summaries.
Clearing the browser cache resolves most rendering issues after a deploy.

Import errors: CSV imports fail loudly on the first bad row with the row
number. Common causes are dates not in ISO format (YYYY-MM-DD), emails missing
the @ sign, and rows with more columns than the header. Fix the reported row
and re-run the import; already-imported rows are skipped, so re-runs are safe.
