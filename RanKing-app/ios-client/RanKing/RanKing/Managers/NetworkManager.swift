//
//  NetworkManager.swift
//  RanKing
//
//  Created by Jonathan Da Silva on 10/13/25.
//

import Foundation
import SwiftUI

enum NetworkError: Error {
    case invalidURL
    case invalidResponse
    case decodingError
}

@MainActor 
class NetworkManager: ObservableObject {
    
    private static var instance: NetworkManager?
    
    public static func getInstance() -> NetworkManager {
        if instance == nil {
            instance = NetworkManager()
        }
        return instance!
    }
        
    // MARK: - GET Request
    func fetchContests(query: String? = "") async throws -> [Contest]? {
        //TODO add query if necessary
        guard let url = URL(string: "https://b5xfrkkof2.execute-api.us-east-2.amazonaws.com/test/contests") else {
            throw NetworkError.invalidURL
        }
        let (data, response) = try await URLSession.shared.data(from: url)
        print(String(data: data, encoding: .utf8) ?? "No data")

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw NetworkError.invalidResponse
        }
        
        let decoder = JSONDecoder()

//         setup date formatter
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)

        decoder.dateDecodingStrategy = .formatted(formatter)
        
        print("200!")
        var contestsRes: [Contest]? = nil
        do {
            let response = try decoder.decode(ContestResponse.self, from: data)
            let contestsData = Data(response.body.utf8)
            contestsRes = try decoder.decode([Contest].self, from: contestsData)
        } catch {
            print("Decoding Error :( \(error)")
            throw NetworkError.decodingError
        }
        
        print("DECODED!!! \(contestsRes ?? [])")
        return contestsRes
    }
    
    func fetchSumbissions(contestId: String) async throws -> [Submission]? {
        print(contestId)
        guard let url = URL(string: "https://b5xfrkkof2.execute-api.us-east-2.amazonaws.com/Deploy1/contests/\(contestId)/submissions") else {
            throw NetworkError.invalidURL
        }
        let (data, response) = try await URLSession.shared.data(from: url)
        print(String(data: data, encoding: .utf8) ?? "No data")
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw NetworkError.invalidResponse
        }
        
        
        var submissions : [Submission]? = nil
        do {
            submissions = try JSONDecoder().decode([Submission].self, from: data)
        } catch {
            print("\(error)")
            throw NetworkError.decodingError
        }
        return submissions
    }

    // MARK: - POST Request
    func register(username: String, email: String, password: String) async throws -> Bool? {
        guard let url = URL(string: "https://b5xfrkkof2.execute-api.us-east-2.amazonaws.com/Deploy1/register") else {
            throw NetworkError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let userData = try JSONSerialization.data(withJSONObject: ["username": username, "email": email, "password_hash": password], options: [])

        request.httpBody = userData

        let (data, response) = try await URLSession.shared.data(for: request)
        print(String(data: data, encoding: .utf8) ?? "No data")
        
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            return false;
        }
        
        let result = String(data: data, encoding: .utf8) ?? "No Data"
        
        if (result.contains("Successfully")) {
            return true
        }
        return false
    }
    
    func login(email: String, password: String) async throws -> String? {
        print("login called")
        guard let url = URL(string: "https://b5xfrkkof2.execute-api.us-east-2.amazonaws.com/Deploy1/login") else {
            throw NetworkError.invalidURL
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let userData = try JSONSerialization.data(withJSONObject: ["email": email, "password_hash": password], options: [])
        request.httpBody = userData
        let (data, response) = try await URLSession.shared.data(for: request)

        print(String(data: data, encoding: .utf8) ?? "No data")
        
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            print("Error on login")
            return nil
        }
        
        var submissions : User? = nil
        do {
            submissions = try JSONDecoder().decode(User.self, from: data)
        } catch {
            print("\(error)")
            throw NetworkError.decodingError
        }
        
        return submissions?.user_id
    }
    
    func sendSubmission(imageData: Data, contestId: Int) async throws -> Bool {
        guard let url = URL(string: "https://b5xfrkkof2.execute-api.us-east-2.amazonaws.com/Deploy1/contests/\(contestId)/submissions") else {
            throw NetworkError.invalidURL
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = createMultipartBody(
            boundary: boundary,
            fieldName: "file",
            fileName: "submission.jpg",
            mimeType: "image/jpeg",
            fileData: imageData
        )
        let (data, response) = try await URLSession.shared.data(for: request)
        print(String(data: data, encoding: .utf8) ?? "No data")
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            return false;
        }
        return true
    }
    
    func getSubmissionImage(submissionId: String) async throws -> Data {
            print("Getting Image for: \(submissionId)")
            
            guard let url = URL(string: "https://b5xfrkkof2.execute-api.us-east-2.amazonaws.com/Deploy1/submissions/\(submissionId)/image") else {
                throw NetworkError.invalidURL
            }

            let (data, response) = try await URLSession.shared.data(from: url)

            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                print("Server error: \((response as? HTTPURLResponse)?.statusCode ?? 0)")
                throw NetworkError.invalidResponse
            }

            // SCENARIO 1: API Gateway already converted it to Binary (Image)
            // If we can make a UIImage directly from the data, we are done.
            if UIImage(data: data) != nil {
                print("Received raw binary image data")
                return data
            }

            // SCENARIO 2: It is a Base64 String
            // We try to convert the data to a UTF8 string
            guard let base64String = String(data: data, encoding: .utf8) else {
                print("Data received was neither an Image nor a String")
                throw NetworkError.decodingError
            }

            // CLEANING:
            // 1. Remove double quotes (if API Gateway sent it as a JSON string "...")
            // 2. Remove newlines or whitespace
            let cleaned = base64String
                .replacingOccurrences(of: "\"", with: "")
                .replacingOccurrences(of: "\n", with: "")
                .replacingOccurrences(of: "\r", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)

            // DECODING:
            // .ignoreUnknownCharacters is the SWIFT equivalent of "base64 -i"
            guard let imageData = Data(base64Encoded: cleaned, options: .ignoreUnknownCharacters) else {
                print("Base64 decode failed. String start: \(cleaned.prefix(20))...")
                throw NetworkError.decodingError
            }
            
            print("Successfully decoded Base64")
            return imageData
        }

    
    
    func sendSubmission2(imageData: Data, contestId: String, userId: String) async throws -> Bool {
        print("Sending image")
        guard let url = URL(string: "https://b5xfrkkof2.execute-api.us-east-2.amazonaws.com/Deploy1/contests/\(contestId)/submissions") else {
            throw NetworkError.invalidURL
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let base64StringImage = imageData.base64EncodedString();
        let userData = try JSONSerialization.data(withJSONObject: ["user_id": userId, "image": base64StringImage, "filename": "example.jpg"], options: [])

        request.httpBody = userData
        
        let (data, response) = try await URLSession.shared.data(for: request)
        print(String(data: data, encoding: .utf8) ?? "No data")
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 201 else {
            return false;
        }
        return true
    }
    
    private func createMultipartBody(
        boundary: String,
        fieldName: String,
        fileName: String,
        mimeType: String,
        fileData: Data
    ) -> Data {
        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"\(fieldName)\"; filename=\"\(fileName)\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: \(mimeType)\r\n\r\n".data(using: .utf8)!)
        body.append(fileData)
        body.append("\r\n".data(using: .utf8)!)
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        return body
    }

    
    func sendVote(submissionId: String, userId : String) async throws -> Bool? {
        guard let url = URL(string: "https://b5xfrkkof2.execute-api.us-east-2.amazonaws.com/Deploy1/submissions/\(submissionId)/vote") else {
            throw NetworkError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let userData = try JSONSerialization.data(withJSONObject: ["user_id": userId], options: [])
        request.httpBody = userData
        
        let (data, response) = try await URLSession.shared.data(for: request)

        print(String(data: data, encoding: .utf8) ?? "No data")
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 201 else {
            return false;
        }
        print("user_id: \(userId) Successfully voted on \(submissionId)")
        return true;
    }
    
    func testServerIsRunning() async throws -> Bool {
        guard let url = URL(string: "http://localhost:8585/") else {
            throw NetworkError.invalidURL
        }

        let (_, response) = try await URLSession.shared.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            print("Connection to server failed.")
            return false
        }
        
        return true
    }
    
    func createContest(name: String, description : String, startDate: String, endDate: String, votingEndDate: String) async throws -> Contest? {
        guard let url = URL(string: "http://localhost/contests") else {
            throw NetworkError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let userData = try JSONSerialization.data(withJSONObject: ["name": name, "description": description, "submission_start_date": startDate, "submission_end_date" : endDate,
                                                                   "voting_end_date" : votingEndDate], options: [])

        request.httpBody = userData

        let (data, response) = try await URLSession.shared.data(for: request)
        print(String(data: data, encoding: .utf8) ?? "No data")

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 201 else {
            throw NetworkError.invalidResponse
        }
        
        
        do {
            let createdPost = try JSONDecoder().decode(Contest.self, from: data)
            return createdPost
        } catch {
            throw NetworkError.decodingError
        }
    }
    
}

extension Data {
    mutating func append(_ string: String) {
        if let data = string.data(using: .utf8) {
            append(data)
        }
    }
}
