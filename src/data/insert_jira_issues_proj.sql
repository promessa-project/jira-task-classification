TRUNCATE TABLE promessa_jira.jira_issues_proj;
INSERT INTO promessa_jira.jira_issues_proj
SELECT
 `project_id`,
  `project_key`,
  `project_name`,
  `summary`,
  `type`,
  `issue`,
  `epic`,
  `labels`,
  `created_at`,
  `clean_summary`,
  `project_status`,
  `epic_summary`,
  `epic_clean_summary`
FROM promessa_jira.jira_issues_data;