# TechCorp API Documentation

## Authentication
All API requests require authentication using Bearer tokens.

POST /api/v1/auth/login
Content-Type: application/json
{
"email": "user@example.com",
"password": "secure_password"
}

Response:
```json
{
  "access_token": "eyJhbGc...",
  "expires_in": 3600
}
```

## User Management Endpoints

### Create User
POST /api/v1/users
Authorization: Bearer {token}
Content-Type: application/json
{
"email": "newuser@example.com",
"name": "John Doe",
"role": "developer"
}

### Get User
GET /api/v1/users/{user_id}
Authorization: Bearer {token}

### Update User
PUT /api/v1/users/{user_id}
Authorization: Bearer {token}
Content-Type: application/json
{
"name": "Jane Doe",
"role": "senior_developer"
}

## Rate Limiting
- 100 requests per minute per API key
- 1000 requests per hour per API key

## Error Codes
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 429: Too Many Requests
- 500: Internal Server Error