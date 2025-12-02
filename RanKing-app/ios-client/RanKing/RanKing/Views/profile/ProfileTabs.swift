//
//  ProfileTabs.swift
//  RanKing
//
//  Created by Damario Hamilton on 11/29/25.
//
import SwiftUI

struct ProfileTabs: View {
    @Binding var selectedTab: Int
    
    var body: some View {
        HStack {
            Button(action: { selectedTab = 0 }) {
                Text("My Submissions")
                    .fontWeight(selectedTab == 0 ? .semibold : .regular)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .background(selectedTab == 0 ? Color.purple.opacity(0.15) : Color.clear)
                    .cornerRadius(10)
            }
            
            Button(action: { selectedTab = 1 }) {
                Text("My Stats")
                    .fontWeight(selectedTab == 1 ? .semibold : .regular)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .background(selectedTab == 1 ? Color.purple.opacity(0.15) : Color.clear)
                    .cornerRadius(10)
            }
        }
        .padding(.horizontal)
    }
}
